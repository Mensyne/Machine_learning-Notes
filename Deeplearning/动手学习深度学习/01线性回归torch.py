import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython import display

### 显示torch 版本
print(torch.__version__)

## 生成1000个样本数据
num_inputs = 2
num_examples = 1000

true_w = [2,-3.4]
true_b = 4.2
features = torch.randn(num_examples,num_inputs,dtypes = torch.float32)
labels = true_w[0]*feature[:0] + true_w[1]*features[:,1] + true_b
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),\
dtypes=troch.float32)

# 使用图像来展示生成的数据
plt.scatter(features[:,1].numpy(),labels.numpy(),1)

## 读取数据集
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices  = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples])
        yield features.index_select(0,j),labels.index_select(0,j)

batch_size = 10
for X ,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break


## 初始化模型参数
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtypes=torch.float32)
b = torch.zeros(1,dtypes = torch.float32)

w.requires_grad_(requires_grad =True)
b.requires_grad_(requires_grad = True)

## 定义模型
def linreg(X,w,b):
    return torch.mm(X,w) + b

## 定义损失函数 均方误差
def squared_loss(y_hat,y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

## 定义损失函数小批量随机梯度下降法
def sgd(params,lr,batch_size):
    for param in params:
        param.data  -= lr.param.grad / batch_size 
    
## 训练
lr = 0.03
num_epochs = 5

net = linreg
loss = squared_loss

# training
for epoch in range(nun_epochs):
    for X,y in data_iter(bacth_size,features,labels):
        l = loss(net(X,w,b).sum()
        l.backward()
        # using small bacth random gradient descent to iter model parameters
        sgd([w,b],lr,batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features,w,b),labels)
    print('epoch %d ,loss %f '%（epoch+1,train_l.mean().item())
print(w,true_w,b,true_b)


## 使用pytorch实现
import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
troch.set_default_tensor_type('troch.FloatTensor')

## 生成数据集
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

## 读取数据集
import torch.utils.data as Data
bacth_size = 10
dataset = Data.TensorDataset(features,labels)

data_iter = Data.DataLoader(

    dataset = dataset,
    bacth_size = batch_size,
    shuffle = True,
    num_workers = 2, ## read data in multithreading
)


for X,y in data_iter:
    print(X,'\n',y)
    break


## 定义模型
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_feature,1)
    
    def forward(self,x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)

## ways to init a multilayer network
## method one
net = nn.Sequential(nn.Linear(num_inputs,1))
## method two
net = nn.Sequential()
net.add_module('linear',nn.Linear(num_inputs,1))
## method three
from  collections import OrderedDict
net = nn.Sequential(OrderedDict(
    [('linear',nn.Linear(num_inputs,1))]
))
print(net)
print(net[0])

## 初始话模型参数
from torch.nn import init
init.normal_(net[0].weight,mean=0.0,std = 0.01)
init.constant_(net[0].bias,val=0.0)
for param in net.parameters():
    print(param)

## 定义损失函数
loss = nn.MSELoss()

## 定义损失函数
import torch.optim as optim

optimizer = optim.SGD(net.parameters(),lr=0.03)
print(optimizer)

## 训练
num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output = net(X)
        l = loss(output,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d，loss:%f'%(epoch,l.item()))

dense = net[0]
print(true_w,dense.weight.data)
print(true_b,dense.bias.data)


