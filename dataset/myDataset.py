#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from torch.utils.data import Dataset


# 导入数据集的类
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.lines = open(csv_file).readlines()

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        cur_line = self.lines[index].split(',')

        # enc_input 具有多个特征
        full_len_input = np.float32([cur_line[0].strip(), cur_line[1].strip()])
        dec_output = np.float32([cur_line[2].strip(), cur_line[3].strip()])
        #enc_input = np.float32(cur_line[0].strip())
        #dec_output = np.float32(cur_line[1].strip())
        return full_len_input, dec_output

    def __len__(self):
        return len(self.lines)  # MyDataSet的行数
