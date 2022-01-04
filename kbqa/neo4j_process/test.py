# -*- coding:utf-8 -*-
__author__ = "lijin"

import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

NEED_DELETE_BEFORE = False
NEED_CREATE = True

raw_data = pd.read_excel(r'neo4j_test_data.xlsx')
graph = Graph(
    host='localhost', port='7687', auth=('neo4j', '124356')
)

if NEED_DELETE_BEFORE:
    graph.delete_all()  # warning!!

node_matcher = NodeMatcher(graph)
relationship_matcher = RelationshipMatcher(graph)

if NEED_CREATE:
    for row in raw_data.itertuples():
        # create node of person
        person = node_matcher.match('Person', id_num=getattr(row, "id_num"),
                                    name=getattr(row, "name")).first()
        if not person:
            person = Node("Person",
                          id_num=getattr(row, "id_num"),
                          name=getattr(row, "name"),
                          bl_1=getattr(row, 'bl_1'),
                          bl_2=getattr(row, 'bl_2'),
                          bl_3=getattr(row, 'bl_3'),
                          bl_4=getattr(row, 'bl_4'),
                          bl_5=getattr(row, 'bl_5'),
                          )
            graph.create(person)

        # create node of phone
        phone = node_matcher.match('Phone', phone_num=getattr(row, "phone")).first()
        if phone is None:
            phone = Node("Phone", phone_num=getattr(row, "phone"))
            graph.create(phone)

        # create relationship of personal_call
        personal_call = relationship_matcher.match(nodes=(person, phone), r_type="personal_call").first()
        if personal_call is None:
            personal_call = Relationship(person, "personal_call", phone)
            graph.create(personal_call)

        # create node of contact_phone_1
        contact_phone_1 = node_matcher.match('Phone', phone_num=getattr(row, "contact_phone_1")).first()
        if contact_phone_1 is None:
            contact_phone_1 = Node("Phone", phone_num=getattr(row, "contact_phone_1"))
            graph.create(contact_phone_1)

        # create relationship of collection_associate_call_1
        collection_associate_call_1 = relationship_matcher.match(nodes=(person, contact_phone_1),
                                                                 r_type="collection_associate_call").first()
        if collection_associate_call_1 is None:
            collection_associate_call_1 = Relationship(person, "collection_associate_call", contact_phone_1)
            graph.create(collection_associate_call_1)

        # create node of contact_phone_2
        contact_phone_2 = node_matcher.match('Phone', phone_num=getattr(row, "contact_phone_2")).first()
        if contact_phone_2 is None:
            contact_phone_2 = Node("Phone", phone_num=getattr(row, "contact_phone_2"))
            graph.create(contact_phone_2)

        # create relationship of collection_associate_call_2
        collection_associate_call_2 = relationship_matcher.match(nodes=(person, contact_phone_2),
                                                                 r_type="collection_associate_call").first()
        if collection_associate_call_2 is None:
            collection_associate_call_2 = Relationship(person, "collection_associate_call", contact_phone_2)
            graph.create(collection_associate_call_2)

        # create node of qq_group_1
        qq_group_1 = node_matcher.match('QQGroup', qq_group_num=getattr(row, "qq_group_1")).first()
        if qq_group_1 is None:
            qq_group_1 = Node("QQGroup", qq_group_num=getattr(row, "qq_group_1"))
            graph.create(qq_group_1)

        # create relationship of collection_qq_group_1
        collection_qq_group_1 = relationship_matcher.match(nodes=(person, qq_group_1),
                                                           r_type="in_collection_qq_group").first()
        if collection_qq_group_1 is None:
            collection_qq_group_1 = Relationship(person, "in_collection_qq_group", qq_group_1)
            graph.create(collection_qq_group_1)

        # create node of qq_group_2
        qq_group_2 = node_matcher.match('QQGroup', qq_group_num=getattr(row, "qq_group_2")).first()
        if qq_group_2 is None:
            qq_group_2 = Node("QQGroup", qq_group_num=getattr(row, "qq_group_2"))
            graph.create(qq_group_2)

        # create relationship of collection_qq_group_2
        collection_qq_group_2 = relationship_matcher.match(nodes=(person, qq_group_2),
                                                           r_type="in_collection_qq_group").first()
        if collection_qq_group_2 is None:
            collection_qq_group_2 = Relationship(person, "in_collection_qq_group", qq_group_2)
            graph.create(collection_qq_group_2)

        # create node of qq_group_3
        qq_group_3 = node_matcher.match('QQGroup', qq_group_num=getattr(row, "qq_group_3")).first()
        if qq_group_3 is None:
            qq_group_3 = Node("QQGroup", qq_group_num=getattr(row, "qq_group_3"))
            graph.create(qq_group_3)

        # create relationship of collection_qq_group_3
        collection_qq_group_3 = relationship_matcher.match(nodes=(person, qq_group_3),
                                                           r_type="in_collection_qq_group").first()
        if collection_qq_group_3 is None:
            collection_qq_group_3 = Relationship(person, "in_collection_qq_group", qq_group_3)
            graph.create(collection_qq_group_3)
