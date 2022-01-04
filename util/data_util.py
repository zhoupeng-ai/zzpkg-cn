import util.common_util as ucu


class SPODataSet:
    r"""SPOData
        SPO数据对象

        参数：
            -**args**： 参数集合

            -**logger**: 日志处理对象

        属性：
            -**len**： 数据长度

        方法：
            get_data(): 返回train_data， val_data， test_data
    """
    def __init__(self, args, data_path, logger, device=ucu.get_device()):
        self.data_path = data_path
        self.device = device
        self.config = args
        self.logger = logger

    def load_spo_data(self):
        import json
        self.logger.info(f'''开始SPO加载数据, 加载路径: {self.data_path}''')
        with open(self.data_path, 'r', encoding='utf-8') as data_file:
            data = json.load(data_file)
        self.logger.info("加载SPO数据结束")

        return data

    def load_spo_schema(self):
        import json

        self.logger.info(f'''开始加载spo_schema数据, 加载路径: {self.config.spo_schema_path}''')
        with open(self.config.spo_schema_path, 'r', encoding='utf-8') as schema_file:
            spo_schema = json.load(schema_file)

        spo_schema_itop = spo_schema[0]
        spo_schema_ptoi = spo_schema[1]
        self.logger.info("加载spo_schema数据结束")

        return spo_schema_itop, spo_schema_ptoi


class SPODataLoader:
    r"""SPODataloader
        获取spo数据加载对象
        参数：
            -**spo_data**： SPOData数据对象
            -**batch_size**： 批处理大小
            -**tokenizer**： Bert分词器
            -**logger**: 日志处理器
    """
    def __init__(self, args, spo_dataset, batch_size, tokenizer, logger):
        self.logger = logger
        self.spo_train_dataset = spo_dataset["spo_train_dataset"]
        self.spo_val_dataset = spo_dataset["spo_val_dataset"]
        self.config = args
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.train_data = self.spo_train_dataset.load_spo_data()
        self.train_spo_schema_itop, self.train_spo_schema_ptoi = self.spo_train_dataset.load_spo_schema()

    def load_batch_data(self):
        return self.data_generator()

    def data_generator(self):
        import numpy as np
        import torch
        texts = []
        self.logger.info(f'''数据加载器处理开始''')
        batch_input_ids, batch_attention_mask = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []

        for i, d in enumerate(self.train_data):
            text = d['text']
            texts.append(text)
            token = self.tokenizer(text=text)
            input_ids, attention_mask = token.input_ids, token.attention_mask
            spo_list = d['spo_list']
            spoes = {}

            for s, p, o in spo_list:
                s_token = self.tokenizer(text=s).input_ids[1:-1]
                o_token = self.tokenizer(text=o).input_ids[1:-1]

                s_start = match(s_token, input_ids)
                o_start = match(o_token, input_ids)
                # self.spo_ptoi = spo_data.spo_schema_ptoi
                # self.spo_itop = spo_data.spo_schema_itop

                p_token = self.train_spo_schema_ptoi[p]

                if s_start != -1 and o_start != -1:
                    s = (s_start, s_start + len(s_token) - 1)
                    o = (o_start, o_start + len(o_token) - 1, p_token)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)

            if spoes:
                # 头、 尾
                s_labels = np.zeros((len(input_ids), 2))
                for s in spoes:
                    s_labels[s[0], 0] = 1
                    s_labels[s[1], 1] = 1
                '''
                print(s_labels.T[0])
                print(s_labels.T[1])
                outputs:len(input_ids) = 10, len(subject) = 4 时的输出
                    [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                    [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
                '''
                start, end = np.array(list(spoes.keys())).T
                # 随机选择一个开始位置
                start = np.random.choice(start)
                # 选择离开始位置最近的结束位置
                end = end[end >= start][0]
                s_ids = (start, end)

                o_labels = np.zeros((len(input_ids), len(self.train_spo_schema_ptoi), 2))
                for o in spoes.get(s_ids, []):
                    o_labels[o[0], o[2], 0] = 1
                    o_labels[o[1], o[2], 1] = 1

                # 构建batch
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_subject_labels.append(s_labels)
                batch_subject_ids.append(s_ids)
                batch_object_labels.append(o_labels)
                if len(batch_subject_labels) == self.batch_size or i == len(self.train_data) - 1:
                    batch_input_ids = sequence_padding(batch_input_ids)
                    batch_attention_mask = sequence_padding(batch_attention_mask)
                    batch_subject_labels = sequence_padding(batch_subject_labels)
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels)
                    yield [
                              torch.from_numpy(batch_input_ids).long(),
                              torch.from_numpy(batch_attention_mask).long(),
                              torch.from_numpy(batch_subject_labels), torch.from_numpy(batch_subject_ids),
                              torch.from_numpy(batch_object_labels)
                          ], None
                    batch_input_ids, batch_attention_mask = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        self.logger.info("数据加载器处理结束")


class Vocab:
    r"""
        构建词典Vocab

        Attribute:
            - **vocab**: 词典
            - **vocab_stoi**: 词典和下标的对应关系

        Inputs:
            - **vocab_path**: 词典的全路径
    """
    def __init__(self, vocab_path):
        self.vocab = Vocab.load_vocab(vocab_path)
        self.vocab_stoi = Vocab.vocab_stoi(self.vocab)

    @staticmethod
    def load_vocab(vocab_path):
        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file.readlines():
                vocab[len(vocab)] = line.strip()
        return vocab

    @staticmethod
    def vocab_stoi(vocab):
        return {word: i for i, word in enumerate(vocab)}

    @property
    def len(self):
        return len(self.vocab)


class SPO(tuple):
    def __init__(self, spo):
        self.spox = (
            spo[0],   # subject
            spo[1],   # predicate
            spo[2],   # object
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def sequence_padding(inputs, padding=0, length=None, mode='post'):
    r"""
        进行序列填充

        Arg:
            - **inputs**: 输入序列列表
            - **padding**: 填充值
            - **length**: 填充长度， 为空时默认填充至序列的最大长度
            - **mode**： 填充模式
                * post: 向后填充
                * pre: 向前填充
    """
    import numpy as np
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def match(pattern, sequence):
    r"""
    从序列sequence 中查找子串pattern, 找到则返回第一个下标， 找不到则返回-1

    Arg:
        - **pattern**: 子串
        - **sequence**: 目标序列
    Return:
        存在时返回第一个下标， 不存在时返回-1
    """
    p_len = len(pattern)
    s_len = len(sequence)
    for i in range(s_len):
        if sequence[i:i+p_len] == pattern:
            return i

    return -1