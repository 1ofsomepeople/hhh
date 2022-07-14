'''
Author: hushuwang hushuwang2019@ia.ac.cn
Date: 2022-07-06 15:58:33
LastEditors: hushuwang hushuwang2019@ia.ac.cn
LastEditTime: 2022-07-13 09:02:56
FilePath: /mahua/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    """

    def __init__(self, file=None, seq_len=None, train=False):
        """
        args:
            file: 训练、测试、验证对应的数据文件
            seq_len: 每一个样本对应的序列长度
            train: 是否参与训练
        """

        data = np.load(file).astype(np.float32)
        _, _, t = data.shape
        self.data = []
        for i in range(t-seq_len+1):
            self.data.append(data[:,:,i:i+seq_len])
        self.data = np.array(self.data)
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        print(self.data.shape)
        self.len = self.data.shape[0]
        if train:
            self.max_val = np.log2(1+data.max())
            print(self.max_val)

    def __len__(self,):
        return self.len

    def __getitem__(self, index):
        return self.data[index][:-1], self.data[index][-1:]

class MyScaler():
    def __init__(self, max_val) -> None:
        self.max_val = max_val

    def transform(self, data):
        return torch.log(data + 1.0) / self.max_val

    def inverse_transform(self, data):
        return torch.exp(data * self.max_val) - 1.0