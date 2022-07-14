'''
Author: hushuwang hushuwang2019@ia.ac.cn
Date: 2022-07-06 14:59:50
LastEditors: hushuwang hushuwang2019@ia.ac.cn
LastEditTime: 2022-07-11 16:55:41
FilePath: /mahua/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np


def log_string(log, string):
    log.write(string + "\n")
    log.flush()
    print(string)



def make_adj(trainset_file, percent=99):
    """
        通过训练集得到的静态邻接矩阵，为每一个节点选出不超过percent的出边、入边
        percent: 0~100
    """
    data = np.load(trainset_file)
    sum_adj = np.sum(data, axis=-1)
    row_threshold = np.percentile(sum_adj, percent, axis=1, keepdims=True)
    col_threshold = np.percentile(sum_adj, percent, axis=0, keepdims=True)
    return np.logical_or(sum_adj>row_threshold, sum_adj>col_threshold).astype(np.float32)