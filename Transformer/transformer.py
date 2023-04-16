#!/usr/bin/env python3
# encoding: utf-8

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化 Shape 为 (max_len, d_model) 的 PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个 tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 这里就是 sin 和 cos 括号中的内容，通过 e 和 ln 进行了变换
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算 PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算 PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 为了方便计算，在最外面在 unsqueeze 出一个 batch
        # pe: [max_len, d_model] -> [batch_size=1, max_len, d_model]
        #pe = pe.unsqueeze(0)
        # 但是我不需要这个 batch_size 的话就不用在前面插入一个维度了

        # 如果一个参数不参与梯度下降，但又希望保存 model 的时候将其保存下来
        # 这个时候就可以用 register_buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [enc_seq_len, enc_feature_size]
        # pe: [max_len, d_model]
        x = x + self.pe[: x.size(0)].requires_grad_(False)
        return self.dropout(x)


class TransformerTS(nn.Module):
    def __init__(self,
                 enc_feature_size,
                 dec_feature_size,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 custom_encoder=None,
                 custom_decoder=None):
        super(TransformerTS, self).__init__()
        # src: (S, E) for unbatched input, (S, N, E) if batch_first=False or (N, S, E) if batch_first=True.
        # tgt: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.
        # output: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.
        # where S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        self.transform = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder
        )
        self.pos = PositionalEncoding(d_model)
        self.enc_input_fc = nn.Linear(enc_feature_size, d_model)
        self.dec_input_fc = nn.Linear(dec_feature_size, d_model)
        self.out_fc = nn.Linear(d_model, dec_feature_size)

    # model(src, tgt)
    # tgt is the decoder input.
    # tgt is short for “target”
    # but this is a little misleading as it is not the actual target sequence
    # but a sequence that consists of the last data point of srcand all the data points of the actual target sequence except that last one.
    # This is why people sometimes refer to the trg sequence as being “shifted right”.
    # The length of trg must be equal to the length of the actual target sequence.
    # Example:
    # In a typical training setup, we train the model to predict 4 future weekly ILI ratios from 10 trailing weekly datapoints.
    # That is, given the encoder input (x1, x2, …, x10) and the decoder input (x10, …, x13),
    # the decoder aims to output (x11, …, x14).
    def forward(self, enc_input, dec_input, src_mask, tgt_mask):
        # embed_encoder_input: [enc_seq_len, enc_feature_size] -> [enc_seq_len, d_model]
        embed_encoder_input = self.pos(self.enc_input_fc(enc_input))

        # embed_decoder_input: [dec_seq_len, dec_feature_size] -> [dec_seq_len, d_model]
        embed_decoder_input = self.dec_input_fc(dec_input)

        # transform
        # x: [dec_seq_len, d_model]
        x = self.transform(src=embed_encoder_input,
                           tgt=embed_decoder_input,
                           src_mask=src_mask,
                           tgt_mask=tgt_mask)

        # x: [dec_seq_len, d_model] -> [dec_seq_len, dec_feature_size]
        x = self.out_fc(x)

        return x
