from dataset import KbqaDataset
from kbqa_model.test_cnn import KbqaTextCnn
import pandas as pd
import jieba
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn

jieba.load_userdict("user_dict.txt")

def tokenize(text):
    res = jieba.cut(text)
    return res


def load_vocab():
    vocab = []
    with open('./data/spo_data/vocab.txt', 'r', encoding="utf-8") as file:
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


def train(mode):
    schema_path = './data/spo_data/schemas.json'
    schema, schema_stoi = load_schema(schema_path)
    schema_array = np.array([int(schema_stoi[i]) for i in schema_stoi])
    vocab, vocab_stoi, vocab_size = load_vocab()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn = KbqaTextCnn(vocab_size=vocab_size, embed_size=100, output_size=len(schema), padding_idx=0).to(device)
    if mode == "train":
        dataset = KbqaDataset('./data/spo_data/sptoo_train.csv', vocab_stoi, max_len=10, schema_path=schema_path)
        print(torch.cuda.is_available())
        data = dataset.data
        total = len(data['question'])

        data = TensorDataset(torch.tensor(data['question']),
                             torch.tensor(data['predicate']))
        train_data_loader = DataLoader(dataset=data, batch_size=8, shuffle=True)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optim = torch.optim.Adam(cnn.parameters(), lr=1e-2)
        best_correct_rate = 0
        epoches = 10
        for epoch in range(epoches):
            total_loss = 0.
            train_correct = 0
            correct_rate = 0
            epoch_total = 0
            for batch in train_data_loader:
                text = batch[0].to(device).long()
                label = batch[1].to(device).long()
                optim.zero_grad()
                output = cnn(text)
                loss = loss_fn(output, label)
                total_loss += loss.item()
                epoch_total += len(label)
                loss.backward()
                optim.step()
                train_correct += (output.argmax(1) == label).sum().item()
                print(total_loss)
                correct_rate = train_correct / epoch_total
                print(f'''当前正确率： {correct_rate:.4f}, 截止目前最好效果{best_correct_rate:.4f}''')
                if correct_rate >= best_correct_rate:
                    torch.save(cnn.state_dict(), './pth/cnn_best.pth')
                    best_correct_rate = correct_rate

        print(f'''正确率： {best_correct_rate:.4f}''')

    if mode == "test":
        cnn.load_state_dict(torch.load('./pth/cnn_best.pth'))
        text = "陈玮的爱人是什么？"
        tokenizers = tokenize(text)
        seq = [vocab_stoi[token] for token in list(tokenizers)]
        seq_padding = padding_seq(seq, 10, vocab_stoi['PAD'])
        seq_data = torch.tensor(seq_padding).unsqueeze(0).to(device).long() # batch_size = 1
        output = cnn(seq_data)
        pred = output.argmax(1).item()
        print(schema[str(pred)])


def padding_seq(seqs, max_len, padding_idx):
    return np.array(
        np.concatenate(
            [seqs, [padding_idx] * (max_len - len(seqs))])
        if len(seqs) < max_len
        else
        seqs[:max_len]
    )


if __name__ == "__main__":
    train("train")

