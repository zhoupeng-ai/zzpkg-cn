from py2neo import Node, Relationship, Graph, NodeMatcher, Subgraph
from tqdm import tqdm
from util.data_util import SPODataSet

from util.common_util import (
    get_device,
    get_logger,
    splice_path
)
import os
from neo4j_graph.config import args
import numpy as np

logger = get_logger(filename=splice_path(args.neo_log, 'import_data.log'))

graph = Graph("http://127.0.0.1:7474", auth=("neo4j", "zzp_kgcn"))
# 清除数据
graph.delete_all()
train_dataset = SPODataSet(args, args.spo_train_path, logger=logger)
val_dataset = SPODataSet(args, args.spo_dev_path, logger=logger)


train_data = train_dataset.load_spo_data()
valid_data = val_dataset.load_spo_data()
train_data.extend(valid_data)

node = 0
relations = 0
logger.info(f'''开始读取数据''')
matcher = NodeMatcher(graph)
tqbar = tqdm(train_data, total=len(train_data))
for i, data in enumerate(tqbar):
    for spo in data["spo_list"]:
        s_node = Node('node', name=spo[0])
        o_node = Node('node', name=spo[2])
        s_m = matcher.match('node').where(f'''_.name=\'{spo[0]}\'''').first()
        o_m = matcher.match('node').where(f'''_.name=\'{spo[2]}\'''').first()
        if s_m is not None:
            graph.create(s_node)
            node += 1
        if o_m is not None:
            graph.create(o_node)
            node += 1
        r = Relationship(s_node, spo[1], o_node)
        graph.create(r)

        relations += 1


logger.info(f'''共需创建节点 {node} 个''')
logger.info(f'''共需创建关系 {relations} 对''')






