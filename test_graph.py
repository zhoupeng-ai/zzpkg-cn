import torch
from util.config import args
from util.common_util import get_device, get_logger
import os
from bert import bert_processor
import numpy as np
from util.data_util import SPO, SPODataSet, SPODataLoader, Vocab
best_model = torch.load(os.path.join(args.model_root_path, 'best_graph_model.bin')).to(get_device())
import json

with open(args.spo_schema_path, 'r', encoding='utf-8') as schema_file:
    spo_schema = json.load(schema_file)

    spo_schema_itop = spo_schema[0]
    spo_schema_ptoi = spo_schema[1]
text = "姚明的妻子是叶莉， 叶莉的丈夫身高是223cm"
vocab = Vocab(os.path.join(args.bert_path, 'vocab.txt')).vocab
logger = get_logger(filename=os.path.join(args.save_path, 'train.log'))
bert_config, bert_tokenizer, bert_model = bert_processor(args, logger, is_fast_token=True)
text_token = bert_tokenizer(text=text, return_tensors='pt')
input_ids, attention_mask = text_token.input_ids.to(get_device()), \
                            text_token.attention_mask.to(get_device())
subject_model = best_model.encoder
_, subject_output = subject_model(input_ids, attention_mask)
subject_pred = subject_output.cpu().data.numpy()
start = np.where(subject_pred[0, :, 0] > 0.9)[0]
end = np.where(subject_pred[0, :, 1] > 0.9)[0]
subjects = []
spo = []
for i in start:
    j = end[end >= i]
    if len(j) > 0:
        j = j[0]
        subjects.append((i, j))
if subjects:
    for s in subjects:
        idx = input_ids.cpu().data.numpy().squeeze(0)[s[0]:s[1] + 1]
        subject = ''.join([vocab[i] for i in idx])

        subject_out, object_out = best_model(input_ids,
                                             attention_mask,
                                             torch.from_numpy(np.array([s])).float().to(get_device()))

        object_preds = object_out.cpu().data.numpy()
        for object_pred in object_preds:
            o_start = np.where(object_pred[:, :, 0] > 0.2)
            o_end = np.where(object_pred[:, :, 1] > 0.2)

            for _start, pred1 in zip(*o_start):
                for _end, pred2 in zip(*o_end):
                    if _start < _end and pred1 == pred2:
                        ids = input_ids.cpu().data.numpy().squeeze(0)[_start:_end + 1]
                        _object = ''.join([vocab[i] for i in ids])
                        predicate = spo_schema_itop[str(pred1)]
                        spo.append([subject, predicate, _object])

print(spo)