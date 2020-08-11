#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:25:32 2020

@author: mensyne
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#定义float类型的Tensor
dtype = torch.FloatTensor

char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
#enumerate() 函数用于将一个可遍历的对象组合成为一个索引序列，
# 这里enumerate的作用是将['a','b','v'……] 变成一个{'a':0,'b':1,'c':2,……}的字典
#即从字符到自然数的索引
word_dict = {n: i for i, n in enumerate(char_arr)}
#number_dict是一个从自然数到字符的索引
number_dict = {i: w for i, w in enumerate(char_arr)}
# number of class(=number of vocab) 字典的条目数
n_class = len(word_dict)

#数据集
seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

#TextLSTM网络的参数
#   步数是3，代表输入为四个字符
#   隐藏层单元数目为128，此为隐藏层中的隐藏单元（特征）数量
n_step = 3
n_hidden = 128

def make_batch(seq_data):
    input_batch, target_batch = [], []

    for seq in seq_data:
        #将数据集分成输入数据和目标字符两部分
        #将seq中的字符转成数字索引存储
        input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    #将input_batch 和 output_batch转换成 Variable形式
    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        #初始化LSTM模型且随机初始化参数W，b
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, X):#定义前向传播函数
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        hidden_state = Variable(torch.zeros(1, len(X), n_hidden))   # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1, len(X), n_hidden))     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model

#构建输入数据和目标数据
input_batch, target_batch = make_batch(seq_data)

model = TextLSTM()

#设定损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()
#Adam算法是一个利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率的优化算法
optimizer = optim.Adam(model.parameters(), lr=0.001)

output = model.forward(input_batch)

# Training
for epoch in range(1000):
    optimizer.zero_grad()

    output = model.forward(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    #反向传播及优化
    loss.backward()
    optimizer.step()

inputs = [sen[:3] for sen in seq_data]

predict = model.forward(input_batch).data.max(1, keepdim=True)[1]
print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])


    
    
        
                
        
        




