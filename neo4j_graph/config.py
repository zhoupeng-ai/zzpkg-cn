import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--neo_log', help='data root', default='')
parser.add_argument('--data_root', help='data root', default='data')
parser.add_argument('--spo_train_path', help='spo training file', default='../data/spo_data/train.json')
parser.add_argument('--spo_dev_path', help='spo validation file', default='../data/spo_data/dev.json')
parser.add_argument('--spo_schema_path', help='spo schemas file', default='../data/spo_data/schemas.json')
parser.add_argument('--spo_vocab_file', help='spo validation file', default='../data/spo_data/vocab.json')
args = parser.parse_args()


