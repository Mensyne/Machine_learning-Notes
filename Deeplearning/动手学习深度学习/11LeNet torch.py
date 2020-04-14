
"""
使用全连接层的局限性
图像在同一列邻近的像素在这个向量中可能相距较远 他们构成的模式可能难以被模式识别
对于大尺寸的输入图像，使用全连接层容易导致模型过大
使用卷积层的优势：
卷积层保留输入形状
卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算 从而避免参数尺寸过大
"""

## LeNet模型
"""
卷积层块里的基本单位是卷积层后接平均池化层：
卷积层用来识别图像里的空间模式，如线条和物体局部，之后的平均池化层则用来降低卷积层对位置的敏感性。
卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用5*5的窗口，
并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。
全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。
"""
import sys
sys.path.append("/home/kesci/input")
import d2lzh1981 as d2l
import torch
import torch.nn as nn
import torch.optim as optim
import time

class Flatten(torch.nn.Module): ## 展平操作
    def forward(self,x):
        return x.view(x.shape[0],-1)

class Reshape(torch.nn.Module): ## 将图像大小重定型
    def forward(self,x): 
        return x.view(-1,1,28,28) #(B x C x H x W)
    
net = torch.nn.Sequential(

    Reshape(),
    nn.Conv2d(in_channels =1,out_channels = 6,kernel_size=5,padding =2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size = 2,stride =2),
    nn.Conv2d(in_channels = 6,out_channels = 16,kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride = 2),
    Flatten(),
    nn.Linear(in_features = 16*5*5,out_features = 120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.Sigmoid(),
    nn.Linear(84,10)
)
# 接下来我们构造一个高和宽均为28的单通道数据样本，并逐层进行前向计算来查看每个层的输出形状。
X = torch.randn(size=(1,1,28,28), dtype = torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)


## 导入数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(
    batch_size=batch_size, root='/home/kesci/input/FashionMNIST2065')
print(len(train_iter))
##为了使读者更加形象的看到数据，添加额外的部分来展示数据的图像
import  matplotlib.pyplot as plt
def show_fashion_mnist(images,labels):
    d2l.use_avg_display()
    _,figs = plt.subplots(1,len(images),figsize=(12,12))
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

for Xdata,ylabel in train_iter:
    break

X,y = [],[]
for i in range(10):
    print(Xdata[i].shape,ylabel[i].numpy())
     X.append(Xdata[i]) # 将第i个feature加到X中
    y.append(ylabel[i].numpy()) # 将第i个label加到y中
show_fashion_mnist(X, y)

## 使用GPU
def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

device = try_gpu()
device

## 计算准确率
def evaluate_accuracy(data_iter, net,device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum,n = torch.tensor([0],dtype=torch.float32,device=device),0
    for X,y in data_iter:
        # If device is the GPU, copy the data to the GPU.
        X,y = X.to(device),y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))  #[[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            n += y.shape[0]
    return acc_sum.item()/n

## 来训练模型
def train_ch5(net, train_iter, test_iter,criterion, num_epochs, batch_size, device,lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            net.train()
            
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device) 
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net,device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc,
                 time.time() - start))

# 训练
lr, num_epochs = 0.9, 10

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
net = net.to(device)

criterion = nn.CrossEntropyLoss()   #交叉熵描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近
train_ch5(net, train_iter, test_iter, criterion,num_epochs, batch_size,device, lr)

for testdata,testlabe in test_iter:
    testdata,testlabe = testdata.to(device),testlabe.to(device)
    break
print(testdata.shape,testlabe.shape)
net.eval()
y_pre = net(testdata)
print(torch.argmax(y_pre,dim=1)[:10])
print(testlabe[:10])