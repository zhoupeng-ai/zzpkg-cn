import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--bert_path', help='config file', default='bert/chinese-bert-wwm')
parser.add_argument('--save_path', help='training log', default='train')
parser.add_argument('--data_root', help='data root', default='data')
parser.add_argument('--spo_train_path', help='spo training file', default='data/spo_data/train.json')
parser.add_argument('--spo_dev_path', help='spo validation file', default='data/spo_data/dev.json')
parser.add_argument('--spo_schema_path', help='spo schemas file', default='data/spo_data/schemas.json')
parser.add_argument('--spo_vocab_file', help='spo validation file', default='data/spo_data/vocab.json')
parser.add_argument('--model_root_path', help='model root path', default='pth')
parser.add_argument('--seed', help='model init seed parameter', type=int, default=2021)
parser.add_argument('--batch_size', help='model train batch size', type=int, default=24)
parser.add_argument('--token_max_length', help='model tokenizer max length', type=int, default=90)
parser.add_argument('--learning_rate', help='model learning rate', type=int, default=5e-5)
parser.add_argument('--batch_split', type=int, default=3)
args = parser.parse_args()


