def is_use_cuda():
    """
        判断GPU是否可用
    """
    import torch

    return torch.cuda.is_available()


def get_device():
    """
        获取GPU或者CPU
    """
    return "cuda" if is_use_cuda() else "cpu"


def init_seed(seed):
    """
       设定随机种子，保证模型可以复现
    """
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if is_use_cuda():
        torch.cuda.manual_seed(seed)


def calculate_acc(pred_y, real_y):
    import torch

    round_y = torch.round(torch.sigmoid(pred_y))

    correct = (round_y == real_y).float()

    acc = correct.sum() / len(correct)

    return acc


def splice_path(root, path):
    r"""
        拼接文件名

        Arg:
            - **root**: 目录
            - **path**: 文件名
        Return:
            全路径
    """
    import os

    return os.path.join(root, path)


# 定义类存储日志信息
def get_logger(filename, print2screen=True):
    import logging

    # 创建一个logger
    logger = logging.getLogger(filename)
    # 设置logger级别为INFO
    logger.setLevel(logging.INFO)
    # 将日志信息输出到磁盘文件上
    fh = logging.FileHandler(filename)
    # 输出到file的logging等级的开关
    fh.setLevel(logging.INFO)
    # 日志信息会输出到指定的stream中，如果stream为空则默认输出到sys.stderr
    ch = logging.StreamHandler()
    # 输出到sys.stderr的logging等级的开关
    ch.setLevel(logging.INFO)
    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s]     '
                                  '==>>    %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将logger添加到handler里面
    logger.addHandler(fh)
    # 如果需要输出到控制台，就将ch加入，并返回
    if print2screen:
        logger.addHandler(ch)
    return logger


