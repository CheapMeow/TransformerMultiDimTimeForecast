#!/usr/bin/env python3
# encoding: utf-8

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from transformer import TimeSeriesTransformer

import utils

# ===============================

# hyper parameters

enc_features_size = 2
dec_features_size = 2
batch_first = False
d_model = 128  # After embedding, the feature length of each element
num_encoder_layers = 4  # Number of encoder layers
num_decoder_layers = 4  # Number of decoder layers
n_heads = 8  # Number of attention heads
dropout_encoder = 0.2
dropout_decoder = 0.2
dropout_pos_enc = 0.1
dim_feedforward_encoder = 1024  # Dimensions of Feed Forward Layer in Encoder
dim_feedforward_decoder = 1024  # Dimensions of Feed Forward Layer in Decoder

lr = 0.001  # learning rate

# ===============================

# Parameters determined by the task situation

# forecast time length

pred_len_size = 10
name_flag = 'Time10'

# batch size

enc_seq_len = pred_len_size
dec_seq_len = pred_len_size

full_seq_len = enc_seq_len + dec_seq_len

# Parameters related to the training situation

total_epoch = 1000  # total epoch
debug_epoch = 10  # Print the situation every debug_epoch

train_loss_list = []  # Save loss every debug_epoch
total_loss = 31433357277  # The largest loss during network training

start_time = 0

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

# ===============================

# 导入数据

from dataset.myDataset import MyDataset
filename = '../data/myData.csv'
dataset_train = MyDataset(filename)
train_loader = DataLoader(dataset_train, batch_size=full_seq_len, shuffle=False, drop_last=True)

# ===============================

# 模型，损失函数，优化器

model = TimeSeriesTransformer(enc_features_size=enc_features_size,
                 dec_features_size=dec_features_size,
                 batch_first=batch_first,
                 d_model=d_model,
                 num_encoder_layers=num_encoder_layers,
                 num_decoder_layers=num_decoder_layers,
                 n_heads=n_heads,
                 dropout_encoder=dropout_encoder,
                 dropout_decoder=dropout_decoder,
                 dropout_pos_enc=dropout_pos_enc,
                 dim_feedforward_encoder=dim_feedforward_encoder,
                 dim_feedforward_decoder=dim_feedforward_decoder).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()

# Make src mask for decoder with size:
# [batch_size*n_heads, dec_seq_len, enc_seq_len]?
# [batch_size*n_heads, enc_seq_len, enc_seq_len]?
src_mask = utils.generate_square_subsequent_mask(
    dim1=dec_seq_len,
    dim2=enc_seq_len
    ).to(device)

# Make tgt mask for decoder with size:
# [batch_size*n_heads, dec_seq_len, dec_seq_len]
tgt_mask = utils.generate_square_subsequent_mask(
    dim1=dec_seq_len,
    dim2=dec_seq_len
    ).to(device)

# ===============================

# 模型训练

def train_transformer(epoch):
    global total_loss
    global start_time
    mode = True
    model.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和
    flag_stop = 0  # 认为收敛的次数
    for idx, (full_len_input, dec_output) in enumerate(train_loader):
        enc_input = full_len_input.numpy()[:enc_seq_len]
        enc_input = Variable(torch.from_numpy(enc_input)).to(device)  # [enc_seq_len, enc_features_size]

        dec_input = dec_output.numpy()[enc_seq_len - 1:enc_seq_len - 1 + dec_seq_len]
        dec_input = Variable(torch.from_numpy(dec_input)).to(device)  # [dec_seq_len, dec_features_size]

        dec_output = dec_output.numpy()[enc_seq_len:enc_seq_len + dec_seq_len]
        dec_output = Variable(torch.from_numpy(dec_output)).to(device)  # [dec_seq_len, dec_features_size]

        # 我在我的电脑上运行的时候到这里就没有问题了
        # 但是在服务器上运行的时候就会报错
        # File "/public/home/.../python3/lib/python3.8/site-packages/torch/nn/modules/transformer.py", line 134, in forward
        # if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
        # IndexError: Dimension out of range(expected to be in range of[-2, 1], but got 2)
        # 这看上去像是我必须在第 2 个维度上等于 d_model
        # 所以我又给输入输出在第 0 个维度上加上了一个 1

        # [enc_seq_len, enc_features_size] -> [enc_seq_len, 1, enc_features_size]
        enc_input = torch.unsqueeze(enc_input, 1)
        # [dec_seq_len, dec_features_size] -> [dec_seq_len, 1, dec_features_size]
        dec_input = torch.unsqueeze(dec_input, 1)

        # [dec_seq_len, 1, dec_features_size]
        prediction = model(enc_input, dec_input, src_mask, tgt_mask)

        # [dec_seq_len, 1, dec_features_size] -> [dec_seq_len, dec_features_size]
        prediction = torch.squeeze(prediction, 1)

        loss = criterion(prediction, dec_output)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        #scheduler.step()
        #print(scheduler.get_lr())

        loss_epoch += loss.item()  # 将每个 batch 的 loss 累加，直到所有数据都计算完毕

        # 如果认为收敛次数大于某个值，就结束
        if loss.item() < 0.05:
            flag_stop += 1
            if flag_stop >= 100:
                break

    if epoch % debug_epoch == 0:
        print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
        if epoch != 0:
            end_time = time.time()
            print("Used time in last {} epochs is {} s".format(debug_epoch, end_time - start_time))
            start_time = end_time
        train_loss_list.append(loss_epoch)
        if loss_epoch < total_loss:  # 损失达到新的最小值时保存模型
            total_loss = loss_epoch
            # 这里也注意要改！
            torch.save(model, '../model/myModel.pkl')  # save model


if __name__ == '__main__':
    # 模型训练
    print("Start Training...")
    start_time = time.time()
    for i in range(total_epoch):  # 模型训练1000轮
        train_transformer(i)
    print("Stop Training!")
