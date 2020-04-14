
## 二维互相关
import torch 
import torch.nn as nn

def corr2d(X, K):
    H, W = X.shape
    h, w = K.shape
    Y = torch.zeros(H - h + 1, W - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
Y = corr2d(X, K)
print(Y)

## 二维卷积层
#二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏置来得到输出。卷积层的模型参数包括卷积核和标量偏置
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

## 例子
X = torch.ones(6, 8)
Y = torch.zeros(6, 7)
X[:, 2: 6] = 0
Y[:, 1] = 1
Y[:, 5] = -1
print(X)
print(Y)

conv2d = Conv2D(kernel_size =(1,2))
step = 30
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y)**2).sum()
    l.backward()
    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
        # 梯度清零
    conv2d.weight.grad.zero_()
    conv2d.bias.grad.zero_()
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

print(conv2d.weight.data)
print(conv2d.bias.data)

## 将核数组上下翻转，左右翻转 在与输入组做互相关运算 这一过程叫做卷积运算

## 特征图与感受野定义
"""
二维卷积层输出的二维数组可以看作是输入在空间维度（宽和高）上某一级的表征，
也叫特征图（feature map）。影响元素的前向计算的所有可能输入区域
(可能大于输入的实际尺寸）叫做的感受野（receptive field）
"""
### 填充和步幅
"""
填充指的是在输入高和宽的两侧填充元素
步幅:卷积和在输入数组上滑动 每次滑动的行数与列数即为步幅
"""

## 多输入通道和多输出通道
"""
我们将大小为3的这一维称为通道维
"""
## 二维卷积层与全连接层对比
"""
二维卷积层常用于处理图像有两个优势
1:全连接层把图像展平成一个向量 在输入图像上相邻的元素可能因为展平操作不再相邻 网络难以捕捉局部信息
而卷积层的设计 天然地具有提取局部信息的能力
2:卷积层的参数量更少
"""
X = torch.rand(4,2,3,5)
print(X.shape)

conv2d = nn.Conv2d(in_channels = 2,out_channels = 3,kernel_size=(3,5),stride = 1,padding = (1,2))
Y = conv2d(X)
print('Y.shape: ', Y.shape)
print('weight.shape: ', conv2d.weight.shape)
print('bias.shape: ', conv2d.bias.shape)

## 池化
"""
池化层主要是计算池化窗口内元素的最大值或平均值
池化层的输出通道道数和输入通道数相等
"""
X = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
pool2d = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))
Y = pool2d(X)
print(X)
print(Y)


