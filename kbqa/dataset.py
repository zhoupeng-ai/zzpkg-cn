import torch
import torch.utils.data as tud
import pandas as pd
import jieba
import numpy as np


class KbqaDataset(tud.Dataset):
    def __init__(self, data_path, vocab, max_len, schema_path):
        super(KbqaDataset, self).__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.data_path = data_path
        self.schema_path = schema_path
        if data_path.endswith(".csv"):
            self.use_pd = True
        else:
            self.use_pd = False

        self.data = self.data_processor()
        self.data_len = 0

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        data = self.data['question']
        return data[index]

    def data_processor(self):
        questions = []
        predicates = []
        schema_itos, schema_stoi = load_schema(self.schema_path)
        if self.use_pd:
            df = pd.read_csv(self.data_path)
            df_question = df['question'].values
            # padding_seq(.apply(seq2index), self.max_len, self.vocab['PAD'])
            df_predicate = df['predicate'].values
            for qu, p in list(zip(df_question, df_predicate)):
                question = seq2index(qu, self.vocab)
                padding = padding_seq(question, self.max_len, self.vocab['PAD'])
                questions.append(padding)
                predicates.append(schema_stoi[p])
            self.data_len = len(df.index)
            return {'question': questions, 'predicate': predicates}
        else:
            with open(self.data_path, 'r', encoding='utf-8') as file:

                for line in file.readlines():
                    line = [li.strip() for li in line.split(',')]
                    question = list(jieba.cut(line[0]))
                    predicate = line[3]
                    if question is not None and predicate is not None:
                        questions.append(question)
                        predicates.append(predicate)
            self.data_len = min(len(questions), len(predicates))
            return {'question': questions, 'answer': predicates}


def tokenizer(text):
    res = jieba.cut(text)
    return res


def seq2index(seq, vocab):
    # _, vocab_stoi, _ = load_vocab()
    tokens = tokenizer(seq)
    return [vocab[token] for token in tokens]


def padding_seq(seqs, max_len, padding_idx):
    return np.array(
        np.concatenate(
            [seqs, [padding_idx] * (max_len - len(seqs))])
        if len(seqs) < max_len
        else
        seqs[:max_len]
    )


def load_vocab():
    vocab = []
    with open('user_dict.txt', 'r', encoding="utf-8") as file:
        for line in file.readlines():
            vocab.append(line.replace("\n", ""))
    vocab_stoi = {vocab[i]: i for i in range(len(vocab))}
    return vocab, vocab_stoi, len(vocab)


def load_schema(schema_path):
    import json
    with open(schema_path, 'r') as load_f:
        json_data = json.load(load_f)
        schema_stoi = json_data['s_to_i']
        schema_itos = json_data['i_to_s']
    return schema_itos, schema_stoi
