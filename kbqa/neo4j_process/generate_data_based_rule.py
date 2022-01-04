from py2neo import Graph, NodeMatcher, RelationshipMatcher
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from neo4j_process.log4neo import get_logger
from sklearn.utils import shuffle
import json
import jieba

logger = get_logger(filename='generate_sp_data.log')


graph = Graph(uri="http://127.0.0.1:7474", auth=("neo4j", "123456"))

sptoo_train_file = "../data/spo_data/sptoo_train.csv"
sotop_train_file = "../data/spo_data/sotop_train.csv"
potos_train_file = "../data/spo_data/potos_train.csv"

sptoo_val_file = "../data/spo_data/sptoo_val.csv"
sotop_val_file = "../data/spo_data/sotop_val.csv"
potos_val_file = "../data/spo_data/potos_val.csv"

relations_matcher = RelationshipMatcher(graph)


# # 根据subject, predicate 得到 object
def generate_sptoo_data():
    logger.info("开始生成数据， 数据格式：（question, answer, subject, predicate, object）")
    df = pd.DataFrame()
    questions = []
    answers = []
    subjects = []
    predicates = []
    objects = []
    labels = []
    user_dict = []
    logger.info("开始生成公司实体数据")
    company_cypher = "match (c:company) return c"
    data = graph.run(company_cypher).to_ndarray()
    for company in tqdm(data):
        c_name = company[0]['name']
        descs = ["是什么？", "是啥？", "是？", "？", "是多少？"]
        predicate_types = ["分红方式", "行业大类", "违规类型", "信用评级", "债券类型", "收入", "收益"]
        predicate_keys = ["assign", "industry", "violations", "dishonesty", "bond", "profit", "profit"]
        for desc in descs:
            for key, value in zip(predicate_keys, predicate_types):
                type_value = company[0][key]
                if type_value is not None and type_value != "":
                    if key == "profit" and desc != descs[-1]:
                        continue
                    if key == "dishonesty":
                        if type_value == -1:
                            type_value = "暂无评级"
                        else:
                            type_value = "优" if type_value == 0 else "差"
                    question = f'''{c_name}的{value}{desc}'''
                    questions.append(question)
                    answers.append(type_value)
                    subjects.append(c_name)
                    predicates.append(value)
                    objects.append(type_value)
                if value not in labels:
                    labels.append(value)
                if value not in user_dict:
                    user_dict.append(value)
        if c_name not in user_dict:
            user_dict.append(c_name)
    logger.info("开始生成人物实体数据")
    person_cypher = "match (p:person) return p"
    data = graph.run(person_cypher).to_ndarray()
    for person in tqdm(data):
        p_name = person[0]['name']
        descs = ["是多少？", "多大？", "是？", "？"]
        for desc in descs:
            age = person[0]['age']
            question = f'''{p_name}的年龄{desc}'''
            questions.append(question)
            answers.append(age)
            subjects.append(p_name)
            predicates.append('年龄')
            objects.append(age)
        if p_name not in user_dict:
            user_dict.append(p_name)
    labels.append("年龄")
    logger.info("开始生成关系对数据")
    relations_cypher = "match ()-[r]->() return r"
    data = graph.run(relations_cypher).data()
    for relation in tqdm(data):
        r = relation.get('r')
        relation_type = type(r).__name__
        subject = r.start_node['name']
        object = r.end_node['name']
        descs = ["是谁？", "是？", "？", "是哪家公司？", "是哪个公司？"]
        for desc in descs:
            if (relation_type == "监事" or relation_type == "董事") and desc in descs[-2:]:
                continue
            question = f'''{subject}的{relation_type}{desc}'''
            questions.append(question)
            answers.append(object)
            subjects.append(subject)
            predicates.append(relation_type)
            objects.append(object)
            if relation_type not in labels:
                labels.append(relation_type)

            if relation_type not in user_dict:
                user_dict.append(relation_type)
    total = len(questions)
    train_len = int(total * 0.8)
    df['question'] = questions
    df['answer'] = answers
    df['subject'] = subjects
    df['predicate'] = predicates
    df['object'] = objects
    logger.info(f'''拆分训练集和验证集: 拆分比例： 0.8''')
    # 打乱数据
    df = shuffle(df)
    train_df = df.iloc[:train_len]

    val_df = df.iloc[train_len:]

    logger.info(f'''Total: {total}, Train-dataset: {train_len}, Val-dataset: {total - train_len}''')
    train_df.to_csv(sptoo_train_file, encoding='utf_8_sig', index=False)
    val_df.to_csv(sptoo_val_file, encoding='utf_8_sig', index=False)

    labels_itos = {i: labels[i] for i in range(len(labels))}
    labels_stoi = {labels[i]: i for i in range(len(labels))}
    data = {"i_to_s": labels_itos, "s_to_i": labels_stoi}

    json_data = json.dumps(data, indent=4)

    with open('../data/spo_data/schemas.json', 'w', encoding="utf-8") as json_file:
        json_file.write(json_data)

    with open('../user_dict.txt', 'w', encoding="utf-8") as txt_file:
        for text in user_dict:
            txt_file.write(text)
            txt_file.write('\n')


def build_vocab(data_path):
    df = pd.read_csv(data_path)
    tokens = df['question'].values
    vocab = []
    vocab.append("PAD")
    vocab.append("UNK")
    with open('../user_dict.txt', 'r', encoding="utf-8") as file:
        for line in file.readlines():
            jieba.add_word(line.replace("\n", ""))
            vocab.append(line.replace("\n", ""))

    with open('../data/spo_data/vocab.txt', 'w', encoding="utf-8") as vocab_file:
        for row in tokens:
            row = tokenize(row)
            for token in list(row):
                if token not in vocab:
                    vocab.append(token)
        for token in vocab:
            vocab_file.write(token)
            vocab_file.write('\n')


def tokenize(text):
    res = jieba.cut(text)
    return res


if __name__ == "__main__":
    generate_sptoo_data()
    build_vocab('../data/spo_data/sptoo_train.csv')