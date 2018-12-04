# coding:utf-8
# @Time :2018/12/2 23:38
# @Author: Mensyne
# @File :softmax.py

import numpy as np

class SoftMax:
    def __init__(self,maxstep=10000,C=1e-4,alpha=0.4):
        self.maxstep = maxstep
        self.C = C # 权值衰减项系数lambda 类似于惩罚系数
        self.alpha = alpha
        self.w  =None # 权值
        self.L = None  # 类的数量
        self.D = None # 输入的样本维度
        self.N = None # 样本总量
    def init_param(self,x_data,y_data):
        '''初始化 暂定输入的数据全部为数值形式'''
        b = np.ones((x_data.shape[0],1))
        x_data = np.hstack((x_data,b))  # 附加偏置项
        self.L = len(np.unique(y_data))
        self.D = x_data.shape[1]
        self.N = x_data.shape[0]
        self.w = np.ones((self.L,self.D)) # l*d 针对每个类 都有一组权值参数w
        return x_data

    def validate(self, x_data, y_data, ratio=0.15):
        N = x_data.shape[0]
        size = int(N * ratio)
        inds = np.random.permutation(range(N))
        for i in range(int(N / size)):
            test_ind = inds[i * size:(i + 1) * size]
            train_ind = list(set(range(N)) - set(test_ind))
            yield x_data[train_ind], y_data[train_ind], x_data[test_ind], y_data[test_ind]

    def bgd(self,x_data,y_data):
        '''梯度下降训练'''
        step =0
        while step <self.maxstep:
            step +=1
            prob = np.exp(x_data @self.w.T) # n*l 行向量存储该样本属于每个类的概率
            nf = np.transpose([prob.sum(axis=1)])
            nf = np.repeat(nf,self.L,axis=1) # n*l
            prob = -prob/nf  # 归一化 此处条件符号仅方便后续计算梯度
            for i in range(self.N):
                prob[i,int(y_data[i])] +=1
            grad = -1.0/self.N*prob.T @x_data+self.C+self.w # 梯度， 第二项为衰减项
            self.w -= self.alpha*grad
        return

    def fit(self,x_data,y_data):
        x_data = self.init_param(x_data,y_data)
        self.bgd(x_data,y_data)

    def predict(self,x):
        b = np.ones((x.shape[0],1))
        x = np.hstack((x,b)) # 附件偏置项
        prob = np.exp(x @ self.w.T)
        return np.argmax(prob,axis=1)

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    data = load_digits()
    x_data = data['data']
    y_data = data['target']
    model = SoftMax(maxstep=100000,alpha=0.1,C=1e-4)
    g = model.validate(x_data,y_data,ratio=0.2)
    for item in g:
        x_train,y_train,x_test,y_test = item
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        score =0
        for y,y_pred in zip(y_test,y_pred):
            score +=1 if y == y_pred else 0
        print(score/len(y_test))





