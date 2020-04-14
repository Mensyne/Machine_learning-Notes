## 从0开始
import torch
import torhcvision
import numpy as np
import sys
sys.path.append('/hom/kesci/input')
import d21zh1981 as d21
print(torch.__version__)
print(torhcvision.__version__)

## 获取训练接和测试集
batch_size = 256
train_iter,test_iter = d21.load_data_fashion_mnist(batch_size)

## 模型参数初始化
num_inputs = 784
print(28*28)
num_inputs = 10
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_inputs)),dtype = torch.float)
b = torch.zeros(num_inputs,dtypes = torch.float)

W.requires_grad_(reuqires_grad = True)
b.requires_grad_(requires_grad = True)

## 对多维tensor 进行按维度操作
X = torch.tensor([[1,2,3],[4,5,6]])
print(X.sum(dim =0,keepdim = True)) # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征

## 定义softmax
def softmax(X):
    X_exp  = X.exp()
    partition = X_exp.sum(dim = 1,keepdim = True)
    return X_exp / partition

X = torch.rand((2,5))
X_prob = softmax(X)
print(X_prob,'\n',X_prob.sum(dim = 1))

## softmax 回归模型
def net(x):
    return softmax(torch.mm(X.view((-1,num_inputs)),W) + b)

## 定义损失函数
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y = torch.LongTensor([0,2])
y_hat.gather(1,y.view(-1,1))

def cross_entroy(y_hat,y):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))

## 定义准确率
def accuracy(y_hat,y):
    return (y_hat.argmax(dim =1) == y).float().mean().item()

def  evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        acc_sum  += (net(X).argmax(dim =1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum /  n 
## 训练模型
num_epochs,lr = 5,0.1

def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,\
        params = None,lr = None,optimizer = None):
        for epoch in range(num_epochs):
            train_1_sum,train_acc_sum ,n = 0.0,0.0,0
            for X,y in train_iter:
                y_hat = net(X)
                l = loss(y_hat,y).sum()

                ## 梯度清零
                if optimizer  is not None:
                    optimizer.zero_grad()
                elif params is not None and params[0].grad is not None:
                    for param in params:
                        param.grad.data.zero_()
                l.backward()
                if optimizer is None:
                    d21.sgd(params,lr,batch_size)
                else:
                    optimizer.step()
                train_1_sum += l.item()
                train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
                n += y.shape[0]
            test_acc = evaluate_accuracy(test_iter,net)
            print('epoch %d,loss %.4f,train acc %.3f,test_acc %.3f'%(epoch + 1,train_1_sum /n,train_acc_sum/n,test_acc))
train_ch3(net,train_iter,test_iter,cross_entroy,num_epochs,batch_size,[W,b],lr)
X,y = iter(test_iter).next()
true_labels = d21.get_fashion_mnist_labels(y.numpy())
pred_labels = d21.get_fashion_mnist_labels(net(X).argmax(dim =1).numpy())
titles = [true + '\n' + pred for true,pred in zip(true_labels,pred_labels)]
d21.show_fashion_mnist(X[0:9],titles[0:9])

## softmax 简介实现
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append('/home/kesci/input')
import d21zh1981 as d21
print(torch.__version__)
batcg_size = 256
train_iter,test_iter = d21.load_data_fashion_mnist(batch_size)
num_inouts = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def forward(self,x):
        y = self.linear(x.view(x.shape[0],-1))
        return y

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)
    
from collections import OrderedDict
net = nn.Sequential(OrderedDict(
    [('flatten',FlattenLayer()),
    ('linear',nn.Linear(num_inputs,num_outputs))]
))

##  初始化模型参数
init.normal_(net.linear.weight,mean = 0,std = 0.01)
init.constant_(net.linear.bias,val = 0)

## 定义损失函数
loss = nn.CrossEntropyLoss()

## 定义优化函数
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)

## 训练
num_epochs = 5
d21.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)









