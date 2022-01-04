from util.data_util import SPO, SPODataSet, SPODataLoader, Vocab
from model import SubjectModel, ObjectModel
from bert import bert_processor
from util.config import args
from util.common_util import (
    get_device,
    init_seed,
    get_logger,
    splice_path
)
import torch
from train import SPOTrainer

import os

init_seed(args.seed)
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

logger = get_logger(filename=splice_path(args.save_path, 'train.log'))

train_path = splice_path(args.save_path, 'train')
log_path = splice_path(args.save_path, 'log')


for path in [train_path, log_path]:
    if not os.path.isdir(path):
        logger.info('cannot find {}, mkdiring'.format(path))
        os.makedirs(path)

for i in vars(args):
    logger.info('{}: {}'.format(i, getattr(args, i)))

logger.info("加载Bert预训练模型")
bert_config, bert_tokenizer, bert_model = bert_processor(args, logger, is_fast_token=True)

logger.info("数据处理开始")
train_dataset = SPODataSet(args, args.spo_train_path, logger=logger)
val_dataset = SPODataSet(args, args.spo_dev_path, logger=logger)
spo_dataset = {"spo_train_dataset": train_dataset,
               "spo_val_dataset": val_dataset,
               "spo_test_dataset": None}
logger.info('数据加载器DataLoader处理')
data_loader = SPODataLoader(args, spo_dataset,
                            batch_size=args.batch_size,
                            tokenizer=bert_tokenizer,
                            logger=logger)
device = get_device()
model_path = splice_path(args.model_root_path, 'graph_model.bin')
if os.path.exists(model_path):
    model = torch.load(model_path).to(device)
    subject_model = model.encoder
else:
    subject_model = SubjectModel.from_pretrained(args.bert_path)
    subject_model = subject_model.to(device)

    object_model = ObjectModel(subject_model)

    model = object_model.to(device)
vocab = Vocab(splice_path(args.bert_path, 'vocab.txt')).vocab
trainer = SPOTrainer(args=args,
                     vocab=vocab,
                     model=[subject_model, model],
                     data_loader=data_loader,
                     spo_dataset=spo_dataset,
                     logger=logger,
                     tokenizer=bert_tokenizer)
trainer.train(epochs=10)

