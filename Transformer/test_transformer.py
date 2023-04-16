#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.append("../../../")

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset.myDataset import MyDataset

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('You are using: ' + str(device))

# batch size
enc_seq_len = 4
dec_seq_len = 3

full_seq_len = enc_seq_len + dec_seq_len

# 导入数据

# 这里注意要根据数据集的名称来改！
filename = '../data/myData.csv'
dataset_test = MyDataset(filename)
test_loader = DataLoader(dataset_test, batch_size=full_seq_len, shuffle=False, drop_last=True)

criterion = nn.MSELoss()  # mean square error


# 测试
def test_transformer():
    # 这里注意要根据模型文件的名称来改！
    net_test = torch.load('../model/myModel.pkl')  # load model
    test_loss = 0
    net_test.eval()
    with torch.no_grad():
        for idx, (full_len_input, dec_output) in enumerate(test_loader):
            enc_input = full_len_input.numpy()[:enc_seq_len]
            enc_input = Variable(torch.from_numpy(enc_input)).to(device)

            dec_input = full_len_input.numpy()[enc_seq_len - 1:enc_seq_len - 1 + dec_seq_len]
            dec_input = Variable(torch.from_numpy(dec_input)).to(device)

            dec_output = full_len_input.numpy()[enc_seq_len:enc_seq_len + dec_seq_len]
            dec_output = Variable(torch.from_numpy(dec_output)).to(device)

            prediction = net_test(enc_input, dec_input)

            print("-------------------------------------------------")
            print("输入:", enc_input)
            print("预期输出:", dec_output)
            print("实际输出:", prediction)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if idx == 0:
                predict_value = prediction
                real_value = dec_output
            else:
                predict_value = torch.cat([predict_value, prediction], dim=0)
                real_value = torch.cat([real_value, dec_output], dim=0)

            loss = criterion(prediction, dec_output)
            test_loss += loss.item()

    print('Test set: Avg. loss: {:.9f}'.format(test_loss))
    return predict_value, real_value


if __name__ == '__main__':
    # 模型测试
    print("testing...")
    p_v, r_v = test_transformer()

    # 对比图
    plt.plot(p_v.cpu(), c='green')
    plt.plot(r_v.cpu(), c='orange', linestyle='--')
    plt.show()
    print("stop testing!")

