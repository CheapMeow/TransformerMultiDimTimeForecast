#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.append("../../../")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset.myDataset import MyDataset
from Transformer.transformer import TransformerTS
import utils

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

# batch size
enc_seq_len = 4
dec_seq_len = 3

full_seq_len = enc_seq_len + dec_seq_len

# total epoch(总共训练多少轮)
total_epoch = 1000

# 1. 导入训练数据
# 这里注意要改
filename = '../data/myData.csv'
dataset_train = MyDataset(filename)
# 从一个 csv 中取数据的时候是一行一行取，然后使用 DataLoader 的话，一行就是一个 batch
# 假设 dataloader 的写法是
# train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=False, drop_last=True)
#
# 对于 NLP，一行有三个元素（三列），一个元素就是一个句子，那么这个元素 split 之后就是 enc_seq_len, dec_seq_len
# 那么训练循环应该写成
# for idx, (enc_input, dec_input, dec_output) in enumerate(train_loader):
# enc_input: [batch_size, enc_seq_len] dec_input: [batch_size, dec_seq_len] dec_output: [batch_size, dec_seq_len]
# 这里的 batch_size 是 DataLoader 中设置的
#
# 但是对于时间预测应用来说，一行是一个时间戳，一行中的一个元素（一列）是一个特征的数值
# 所以对于时间预测应用来说，一行应该是 enc_seq_len, dec_seq_len 中的一个
# 那么训练循环应该写成
# for idx, (full_len_input, dec_output) in enumerate(train_loader):
# full_len_input: [batch_size, enc_feature_size] dec_output: [batch_size, dec_feature_size]
# 所以说 DataLoader 这里的 batch_size
# 与 transformer_model = nn.Transformer(...) transformer_model(src, tgt) 的 src, tgt 的 batch_size 不是一回事
# 不考虑 transformer 的 batch, transformer_model 的输入输出是 src: (S, E) tgt: (T, E) output: (T, E)
# where S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
# 所以 DataLoader 的 batch_size 应该转换成 transformer 的输入输出的 S T
#
# 参考 https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
#
# That is, given the encoder input (x1, x2, …, x10) and the decoder input (x10, …, x13),
# the decoder aims to output (x11, …, x14).
#
# 现在已经有的 full_len_input 相当于 (x1, x2, ..., x14)
# 所以需要 enc_input = full_len_input.numpy()[:enc_seq_len] 获得 (x1, x2, ..., x10)
# dec_input = full_len_input.numpy()[enc_seq_len-1:enc_seq_len-1+dec_seq_len] 获得 (x10, …, x13)
# dec_output = full_len_input.numpy()[enc_seq_len:enc_seq_len+dec_seq_len] 获得 (x11, …, x14)
# 那么其实 full_seq_len = enc_seq_len+dec_seq_len
#
# 而 transformer 的 E 其实就是 d_model
# 所以如果我们要输入一个自定义的 enc_feature_size 输出一个自定义的 dec_feature_size
# 其实就在 transformer 的前后加两个线性层
# 前面的线性层 [:, enc_feature_size] -> [:, d_model]
# 后面的线性层 [:, d_model] -> [:, dec_feature_size]
train_loader = DataLoader(dataset_train, batch_size=full_seq_len, shuffle=False, drop_last=True)

# 2. 构建模型，优化器
# 输入特征维度可能要改
model = TransformerTS(enc_feature_size=2,
                    dec_feature_size=2,
                    d_model=32,  # 编码器/解码器输入中预期特性的数量
                    nhead=8,
                    num_encoder_layers=3,
                    num_decoder_layers=3,
                    dim_feedforward=32,
                    dropout=0.1,
                    activation='relu',
                    custom_encoder=None,
                    custom_decoder=None).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()
train_loss_list = []  # 每次epoch的loss保存起来
total_loss = 31433357277  # 网络训练过程中最大的loss

# Make src mask for decoder with size:
# [batch_size*n_heads, dec_seq_len, enc_seq_len]?
# [batch_size*n_heads, enc_seq_len, enc_seq_len]?
src_mask = utils.generate_square_subsequent_mask(
    dim1=enc_seq_len,
    dim2=enc_seq_len
    )

# Make tgt mask for decoder with size:
# [batch_size*n_heads, dec_seq_len, dec_seq_len]
tgt_mask = utils.generate_square_subsequent_mask(
    dim1=dec_seq_len,
    dim2=dec_seq_len
    )

# 3. 模型训练
def train_transformer(epoch):
    global total_loss
    mode = True
    model.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和
    for idx, (full_len_input, dec_output) in enumerate(train_loader):
        enc_input = full_len_input.numpy()[:enc_seq_len]
        enc_input = Variable(torch.from_numpy(enc_input)).to(device)

        dec_input = full_len_input.numpy()[enc_seq_len - 1:enc_seq_len - 1 + dec_seq_len]
        dec_input = Variable(torch.from_numpy(dec_input)).to(device)

        dec_output = full_len_input.numpy()[enc_seq_len:enc_seq_len + dec_seq_len]
        dec_output = Variable(torch.from_numpy(dec_output)).to(device)

        prediction = model(enc_input, dec_input, src_mask, tgt_mask)
        loss = criterion(prediction, dec_output)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        #scheduler.step()
        #print(scheduler.get_lr())

        loss_epoch += loss.item()  # 将每个 batch 的 loss 累加，直到所有数据都计算完毕
        if epoch % 100 == 0:
            if idx == len(train_loader) - 1:
                print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
                train_loss_list.append(loss_epoch)
                if loss_epoch < total_loss:
                    total_loss = loss_epoch
                    # 这里也注意要改！
                    torch.save(model, '../model/myModel.pkl')  # save model


if __name__ == '__main__':
    # 模型训练
    print("Start Training...")
    for i in range(total_epoch):  # 模型训练1000轮
        train_transformer(i)
    print("Stop Training!")
