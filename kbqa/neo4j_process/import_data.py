from py2neo import Graph, Node, Relationship, Subgraph, NodeMatcher, RelationshipMatcher
from tqdm import tqdm
import pandas as pd
import numpy as np
from neo4j_process.log4neo import get_logger
import os

graph = Graph(uri="http://127.0.0.1:7474", auth=("neo4j", "123456"))
matcher = NodeMatcher(graph)
r_matcher = RelationshipMatcher(graph)
logger = get_logger(filename='neo4j.log')


def import_person(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理人物节点入库， 数据文件：{data_path}''')
    p_total = len(df.index)
    person_name = df["personname"].values
    person_id = df["personcode"].values
    nodes = []
    data = list(zip(person_id, person_name))
    effective_node = 0
    exists_n = 0
    for pid, pname in tqdm(data):
        age = np.random.randint(20, 70, 1)[0]
        p_match = matcher.match('person').where(f'''_.pid=\'{pid}\'''').first()
        if p_match is None:
            node = Node('person', name=pname, age=int(age), pid=str(pid))
            nodes.append(node)
            effective_node += 1
        else:
            exists_n += 1
    if effective_node != 0:
        graph.create(Subgraph(nodes))
    logger.info(f'''共需导入人物节点： {p_total} 个， 有效节点： {effective_node} 个, 已存在节点：{exists_n}个''')


def set_bond_type(data_path):
    df = pd.read_csv(data_path)
    b_total = len(df.index)
    bond_type_itos = list(df['securitytype'])
    bond_type_stoi = {bond_type_itos[i]: i for i in range(b_total)}
    return bond_type_itos, bond_type_stoi


# eid,companyname,dishonesty
def import_company(data_path):
    df = pd.read_csv(data_path)
    c_total = len(df.index)
    logger.info(f'''处理公司节点入库， 数据文件：{data_path}''')
    c_name = df['companyname'].values
    c_id = df['eid'].values
    c_dishonesty = df['dishonesty'].values
    data = list(zip(c_id, c_name, c_dishonesty))
    nodes = []
    exists_n = 0
    for id, name, dishonesty in tqdm(data):
        # 收入
        profit = np.random.randint(100000, 10000000, 1)[0]
        c_match = matcher.match("company").where(f'''_.cid=\'{id}\'''').first()
        if c_match is None:
            node = Node("company", name=name, cid=str(id), dishonesty=int(dishonesty), profit=int(profit))
            nodes.append(node)
        else:
            exists_n += 1
    effective_node = len(nodes)
    if effective_node != 0:
        graph.create(Subgraph(nodes))
    logger.info(f'''共需导入公司节点： {c_total} 个， 有效节点： {effective_node} 个, 已存在节点：{exists_n}个''')


# eid,pid,post
def import_company_r_person(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-人物关系， 数据文件：{data_path}''')
    r_total = len(df.index)
    eid = df['eid'].values
    pid = df['pid'].values
    post = df['post'].values
    data = list(zip(eid, pid, post))
    c_p_relations = []
    exists_r = 0
    for c_id, p_id, c_r_p in tqdm(data):
        person = matcher.match("person").where(f'''_.pid=\'{p_id}\'''').first()
        company = matcher.match("company").where(f'''_.cid=\'{c_id}\'''').first()
        if person is not None and company is not None:
            r_match = r_matcher.match(nodes=(company, person), r_type=c_r_p)
            exists_r += len(list(r_match))
            if len(list(r_match)) == 0:
                relations = Relationship(company, c_r_p, person)
                c_p_relations.append(relations)

    effective_relations = len(c_p_relations)
    if effective_relations != 0:
        graph.create(Subgraph(relationships=c_p_relations))
    logger.info(f'''导入公司-人物关系完成， 共需导入： {r_total} 个， 有效关系： {effective_relations} 个, 已存在关系：{exists_r}个''')


def set_company_person_r(data_path):
    df = pd.read_csv(data_path)
    post_type_itos = list(df['post'].drop_duplicates())
    b_total = len(post_type_itos)
    # print(post_type_itos)
    post_type_stoi = {post_type_itos[i]: i for i in range(b_total)}
    return post_type_itos, post_type_stoi


# eid1,eid2
def import_company_r_supplier(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-供应商关系， 数据文件：{data_path}''')
    r_total = len(df.index)
    eid1 = df['eid1'].values
    eid2 = df['eid2'].values
    data = list(zip(eid1, eid2))
    c_s_relations = []
    exists_r = 0
    for c_id, supplier_id in tqdm(data):
        if c_id == supplier_id:
            continue
        company = matcher.match("company").where(f'''_.cid=\'{c_id}\'''').first()
        supplier = matcher.match("company").where(f'''_.cid=\'{supplier_id}\'''').first()
        c_r_s = "供应商"
        if company is not None and supplier is not None:
            r_match = r_matcher.match(nodes=(company, supplier), r_type=c_r_s)
            exists_r += len(list(r_match))
            if len(list(r_match)) == 0:
                relations = Relationship(company, c_r_s, supplier)
                c_s_relations.append(relations)

    effective_relations = len(c_s_relations)
    if effective_relations != 0:
        graph.create(Subgraph(relationships=c_s_relations))
    logger.info(f'''导入公司-供应商关系完成， 共需导入： {r_total} 个， 有效关系： {effective_relations} 个, 已存在关系：{exists_r}个''')


# bond,eid
def import_company_attr_bond(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-债券属性， 数据文件：{data_path}''')
    r_total = len(df.index)
    bond = df['bond'].values
    eid = df['eid'].values
    data = list(zip(bond, eid))
    nodes = []
    for bond, c_id in tqdm(data):
        companys = matcher.match("company").where(f'''_.cid=\'{c_id}\'''')
        for company in companys:
            if pd.isnull(bond):
                company["bond"] = ""
            else:
                company["bond"] = bond
            nodes.append(company)
    graph.push(Subgraph(nodes))


# assign,eid
def import_company_attr_assign(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-分红方式， 数据文件：{data_path}''')
    r_total = len(df.index)
    assign = df['assign'].values
    eid = df['eid'].values
    data = list(zip(assign, eid))
    nodes = []
    for assign, c_id in tqdm(data):
        companys = matcher.match("company").where(f'''_.cid=\'{c_id}\'''')
        for company in companys:
            if pd.isnull(assign):
                company["assign"] = ""
            else:
                company["assign"] = assign
            nodes.append(company)
    graph.push(Subgraph(nodes))


# eid,dishonesty
def import_company_attr_dishonesty(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-失信情况， 数据文件：{data_path}''')
    r_total = len(df.index)
    dishonesty = df['dishonesty'].values
    eid = df['eid'].values
    data = list(zip(dishonesty, eid))
    nodes = []
    for dishonesty, c_id in tqdm(data):
        companys = matcher.match("company").where(f'''_.cid=\'{c_id}\'''')
        for company in companys:
            if pd.isnull(dishonesty):
                company["dishonesty"] = int("-1")
            else:
                company["dishonesty"] = int(dishonesty)
            nodes.append(company)
    graph.push(Subgraph(nodes))


# eid1,eid2
def import_company_r_customer(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-客户关系， 数据文件：{data_path}''')
    r_total = len(df.index)
    eid1 = df['eid1'].values
    eid2 = df['eid2'].values
    data = list(zip(eid1, eid2))
    c_c_relations = []
    exists_r = 0
    for c_id, customer_id in tqdm(data):
        if c_id == customer_id:
            continue
        company = matcher.match("company").where(f'''_.cid=\'{c_id}\'''').first()
        customer = matcher.match("company").where(f'''_.cid=\'{customer_id}\'''').first()
        c_r_c = "客户"
        if company is not None and customer is not None:
            r_match = r_matcher.match(nodes=(company, customer), r_type=c_r_c)
            exists_r += len(list(r_match))
            if len(list(r_match)) == 0:
                relations = Relationship(company, c_r_c, customer)
                c_c_relations.append(relations)

    effective_relations = len(c_c_relations)
    if effective_relations != 0:
        graph.create(Subgraph(relationships=c_c_relations))
    logger.info(f'''导入公司-供应商关系完成， 共需导入： {r_total} 个， 有效关系： {effective_relations} 个, 已存在关系：{exists_r}个''')


# eid1,eid2
def import_company_r_guaranty(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-担保关系， 数据文件：{data_path}''')
    r_total = len(df.index)
    eid1 = df['eid1'].values
    eid2 = df['eid2'].values
    data = list(zip(eid1, eid2))
    c_g_relations = []
    exists_r = 0
    for c_id, guaranty_id in tqdm(data):
        if c_id == guaranty_id:
            continue
        company = matcher.match("company").where(f'''_.cid=\'{c_id}\'''').first()
        guaranty = matcher.match("company").where(f'''_.cid=\'{guaranty_id}\'''').first()
        c_r_g = "担保"
        if company is not None and guaranty is not None:
            r_match = r_matcher.match(nodes=(company, guaranty), r_type=c_r_g)
            exists_r += len(list(r_match))
            if len(list(r_match)) == 0:
                relations = Relationship(company, c_r_g, guaranty)
                c_g_relations.append(relations)

    effective_relations = len(c_g_relations)
    if effective_relations != 0:
        graph.create(Subgraph(relationships=c_g_relations))
    logger.info(f'''导入公司-供应商关系完成， 共需导入： {r_total} 个， 有效关系： {effective_relations} 个, 已存在关系：{exists_r}个''')


# industry,eid
def import_company_attr_industry(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-所属行业， 数据文件：{data_path}''')
    r_total = len(df.index)
    industry = df['industry'].values
    eid = df['eid'].values
    data = list(zip(industry, eid))
    nodes = []
    for industry, c_id in tqdm(data):
        companys = matcher.match("company").where(f'''_.cid=\'{c_id}\'''')
        for company in companys:
            if pd.isnull(industry):
                company["industry"] = "其他"
            else:
                company["industry"] = industry
            nodes.append(company)
    graph.push(Subgraph(nodes))


# violations,eid
def import_company_attr_violations(data_path):
    df = pd.read_csv(data_path)
    logger.info(f'''处理公司-违规类型， 数据文件：{data_path}''')
    r_total = len(df.index)
    violations = df['violations'].values
    eid = df['eid'].values
    data = list(zip(violations, eid))
    nodes = []
    for violations, c_id in tqdm(data):
        companys = matcher.match("company").where(f'''_.cid=\'{c_id}\'''')
        for company in companys:
            if pd.isnull(violations):
                company["violations"] = "无"
            else:
                company["violations"] = violations
            nodes.append(company)
    graph.push(Subgraph(nodes))


def set_assign_type(data_path):
    df = pd.read_csv(data_path)
    b_total = len(df.index)
    assign_type_itos = list(df['schemetype'])
    assign_type_stoi = {assign_type_itos[i]: i for i in range(b_total)}
    return assign_type_itos, assign_type_stoi


def set_industry_type(data_path):
    df = pd.read_csv(data_path)
    b_total = len(df.index)
    industry_type_itos = list(df['orgtype'])
    industry_type_stoi = {industry_type_itos[i]: i for i in range(b_total)}
    return industry_type_itos, industry_type_stoi


def set_violations_type(data_path):
    df = pd.read_csv(data_path)
    b_total = len(df.index)
    violations_type_itos = list(df['gooltype'])
    violations_type_stoi = {violations_type_itos[i]: i for i in range(b_total)}
    return violations_type_itos, violations_type_stoi


def delete_relation():
    cypher = 'match ()-[r]-() delete r'
    graph.run(cypher)
    logger.info("正在删除关系。。。")


def delete_node():
    cypher = 'match (n) delete n'
    graph.run(cypher)
    logger.info("正在删除节点。。。")


def delete_data():
    delete_relation()
    delete_node()
    logger.info('删除完成')


def import_data(mode=None):
    if mode is None:
        mode = "update"
    logger.info(f'''开始导入数据， 模式：{mode}''')
    if mode == "insert":
        delete_data()
    import_person("../data/人物.csv")
    import_company("../data/公司.csv")
    import_company_r_person("../data/公司-人物.csv")
    import_company_r_supplier("../data/公司-供应商.csv")
    import_company_attr_bond("../data/公司-债券.csv")
    import_company_attr_assign("../data/公司-分红.csv")
    import_company_attr_dishonesty("../data/公司-失信.csv")
    import_company_r_customer("../data/公司-客户.csv")
    import_company_r_guaranty("../data/公司-担保.csv")
    import_company_attr_industry("../data/公司-行业.csv")
    import_company_attr_violations("../data/公司-违规.csv")


def get_data_dict():
    bond_type_itos, bond_type_stoi = set_bond_type('../data/债券类型.csv')
    print(bond_type_itos)
    print(bond_type_stoi)
    post_type_itos, post_type_stoi = set_company_person_r("../data/公司-人物.csv")
    print(post_type_itos, post_type_stoi)
    assign_type_itos, assign_type_stoi = set_assign_type('../data/分红.csv')
    print(assign_type_itos)
    print(assign_type_stoi)
    industry_type_itos, industry_type_stoi = set_industry_type('../data/行业.csv')
    print(industry_type_itos)
    print(industry_type_stoi)
    violations_type_itos, violations_type_stoi = set_violations_type('../data/违规类型.csv')
    print(violations_type_itos)
    print(violations_type_stoi)
    return {
        bond_type_itos: bond_type_itos,
        bond_type_stoi: bond_type_stoi,
        post_type_itos: post_type_itos,
        post_type_stoi: post_type_stoi,
        assign_type_itos: assign_type_itos,
        assign_type_stoi: assign_type_stoi,
        industry_type_itos: industry_type_itos,
        industry_type_stoi: industry_type_stoi,
        violations_type_itos: violations_type_itos,
        violations_type_stoi: violations_type_stoi
    }


def get_data():
    cypher = "match crp=(c:company)-[r:`监事`]->(p:person) return c limit 10"
    result = graph.run(cypher)
    while result.forward():
        print(result[0]['bond'])


if __name__ == "__main__":
    # import_data("insert")
    # dic = get_data_dict()
    get_data()
