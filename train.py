from tqdm import tqdm
import numpy as np
from util.common_util import get_device
import torch.optim as optim
from transformers import AdamW
import torch
import torch.nn as nn
import json
from util.common_util import splice_path
from util.data_util import SPO
from torch.utils.tensorboard import SummaryWriter


class SPOTrainer:
    def __init__(self, args, vocab, spo_dataset, data_loader, tokenizer, logger, model, device=get_device()):
        self.config = args
        self.vocab = vocab
        self.device = device
        self.spo_train_dataset = spo_dataset["spo_train_dataset"]
        self.spo_val_dataset = spo_dataset["spo_val_dataset"]
        self.train_spo_schema_itop, _ = self.spo_train_dataset.load_spo_schema()
        self.spo_train_data = self.spo_train_dataset.load_spo_data()
        self.spo_val_data = self.spo_val_dataset.load_spo_data()
        self.data_loader = data_loader
        self.logger = logger
        self.subject_model, self.model = model[0], model[1]
        self.tokenizer = tokenizer
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.0}
        ]
        self.optim = AdamW(self.optimizer_grouped_parameters, lr=args.learning_rate)
        self.loss_fn = nn.BCELoss()
        self.model_path = splice_path(args.model_root_path, 'graph_model.bin')
        self.best_val_f1 = 0.
        self.train_writer = SummaryWriter(splice_path(args.save_path, 'train_tsb'))
        self.valid_writer = SummaryWriter(splice_path(args.save_path, 'valid_tsb'))
        self.curr_step = 0

    def train(self, epochs):
        self.logger.info("开始训练")
        for epoch in range(epochs):
            epoch_suffix = f'''{epoch}st''' if epoch == 1 \
                else f'''{epoch}nd''' if epoch == 2 \
                else f'''{epoch}rd''' if epoch == 3 \
                else f'''{epoch}th'''
            self.logger.info(f'''Epoch: {epoch_suffix}''')
            self._train(epoch)

    def _train(self, epoch):
        train_loss = 0.
        self.model.train()
        train_data_loader = self.data_loader.load_batch_data()
        tqdm_bar = tqdm(enumerate(train_data_loader), desc='Train (epoch #{})'.format(epoch),
                        dynamic_ncols=True)
        for step, batch in tqdm_bar:
            self.optim.zero_grad()
            self.curr_step += 1
            batch = batch[0]
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            subject_labels = batch[2].to(self.device)
            subject_ids = batch[3].to(self.device)
            object_labels = batch[4].to(self.device)
            subject_out, object_out = self.model(input_ids, attention_mask, subject_ids.float())
            subject_out = subject_out * attention_mask.unsqueeze(-1)
            object_out = object_out * attention_mask.unsqueeze(-1).unsqueeze(-1)
            subject_loss = self.loss_fn(subject_out, subject_labels.float())
            object_loss = self.loss_fn(object_out, object_labels.float())

            loss = subject_loss + object_loss
            train_loss += loss.item()
            full_loss = loss / self.config.batch_split
            full_loss.backward()
            lr = self.optim.param_groups[0]["lr"]
            if (step + 1) % self.config.batch_split == 0:
                self.optim.step()
                self.optim.zero_grad()
                tqdm_bar.set_postfix({'loss': loss.item()})
                self.train_writer.add_scalar('ind/loss', loss, self.curr_step)
                self.train_writer.add_scalar('ind/lr', lr, self.curr_step)

            tqdm_bar.update()
            tqdm_bar.set_description(f'''train loss:{loss.item()}''')

            if step % 100 == 0 and step != 0:
                suffix = f'''{self.curr_step}st''' if self.curr_step == 1 \
                    else f'''{self.curr_step}nd''' if self.curr_step == 2 \
                    else f'''{self.curr_step}rd''' if self.curr_step == 3 \
                    else f'''{self.curr_step}th'''
                self.logger.info(f'''Current Step: {suffix}''')
                self.model.eval()
                f1, precision, recall = self._evaluate(epoch, step)
                self.valid_writer.add_scalar('ind/f1', f1, self.curr_step)
                self.valid_writer.add_scalar('ind/precision', precision, self.curr_step)
                self.valid_writer.add_scalar('ind/recall', recall, self.curr_step)
                if f1 > self.best_val_f1:
                    self.best_val_f1 = f1
                    self.logger.info(f'''best_val_f1: {self.best_val_f1}, Save the best Model''')
                    torch.save(self.model, self.model_path)
                self.model.train()
        self.logger.info(f'''Total loss: {train_loss}''')

    def _eval(self, data):
        val_loss = 0.

        with torch.no_grad():
            spo = []
            text = data['text']
            spo_list = data['spo_list']
            text_token = self.tokenizer(text=text, return_tensors='pt')
            input_ids, attention_mask = text_token.input_ids.to(self.device), \
                                        text_token.attention_mask.to(self.device)
            _, subject_output = self.subject_model(input_ids, attention_mask)
            subject_pred = subject_output.cpu().data.numpy()
            start = np.where(subject_pred[0, :, 0] > 0.6)[0]
            end = np.where(subject_pred[0, :, 1] > 0.5)[0]
            subjects = []
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    subjects.append((i, j))
            if subjects:
                for s in subjects:
                    idx = input_ids.cpu().data.numpy().squeeze(0)[s[0]:s[1] + 1]
                    subject = ''.join([self.vocab[i] for i in idx])

                    subject_out, object_out = self.model(input_ids,
                                                         attention_mask,
                                                         torch.from_numpy(np.array([s])).float().to(self.device))

                    object_preds = object_out.cpu().data.numpy()
                    for object_pred in object_preds:
                        o_start = np.where(object_pred[:, :, 0] > 0.2)
                        o_end = np.where(object_pred[:, :, 1] > 0.2)

                        for _start, pred1 in zip(*o_start):
                            for _end, pred2 in zip(*o_end):
                                if _start < _end and pred1 == pred2:
                                    ids = input_ids.cpu().data.numpy().squeeze(0)[_start:_end + 1]
                                    _object = ''.join([self.vocab[i] for i in ids])
                                    predicate = self.train_spo_schema_itop[str(pred1)]
                                    spo.append([subject, predicate, _object])

            return spo

    def _evaluate(self, epoch, step):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        t_f1, t_precision, t_recall = 0., 0., 0.
        f = open('pth/dev_pred.json', 'w', encoding='utf-8')
        spo_val_data = self.spo_val_data[0:100]
        val_data_len = len(spo_val_data)
        pbar = tqdm(spo_val_data, desc='val loss (epoch #{})'.format(epoch),
                    dynamic_ncols=True)
        f.write('\"' + str(epoch) + '-' + str(step) + '\" : { \n')
        for data in pbar:
            # 预测结果
            text = data['text']
            spo_list = data['spo_list']

            R = set([SPO(_spo) for _spo in self._eval(data)])
            # 真实结果
            T = set([SPO(_spo) for _spo in spo_list])
            # R = set(spo_ori)
            # T = set(spo)
            # 交集
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            s = json.dumps({
                    'text': text,
                    'spo_list': list(T),
                    'spo_list_pred': list(R),
                    'new': list(R - T),
                    'lack': list(T - R),
                },
                ensure_ascii=False,
                indent=4)

            f.write(s + ', \n')
            f.write('}, \n')
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            t_f1 += f1
            t_recall += recall
            t_precision += precision
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )

        pbar.close()
        f.close()
        self.logger.info(f'''epoch: {epoch}, step: {step}, f1: {t_f1/val_data_len}, 
                                                            precision: {t_precision/val_data_len}, 
                                                            recall: {t_recall/val_data_len}''')
        return t_f1/val_data_len, t_precision/val_data_len, t_recall/val_data_len
