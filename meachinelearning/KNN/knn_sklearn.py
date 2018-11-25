# coding:utf-8
# @Time :2018/11/24 0:12
# @Author: Mensyne
# @File :knn_sklearn.py

from sklearn.model_selection import train_test_split
from collections import Counter
from numpy.linalg import norm

import numpy as np
import matplotlib.pyplot as plt


class KNN():
    '''
    使用了两种距离来计算 一种是余弦距离 一种是欧式距离
    :param self.k = kwargs.pop('k',1)
    :param self.distance = kwargs.pop('distance','cosin')
    :param self.train_data = kwargs.pop('train_data',None)
    :param self.train_label = kwargs.pop('train_label',None)
    :param self.test_data = kwargs.pop('test_data',None)
    :param self.test_label
    :param self.neighbor_data
    :param self.neighbor_label
    '''

    def __init__(self,**kwargs):
        self.k = kwargs.pop('k',1)
        self.distance = kwargs.pop('distance','euclidean')
        self.train_data = kwargs.pop('train_data',None)
        self.train_label = kwargs.pop('train_label',None)
        self.test_data = kwargs.pop('test_data',None)
        if kwargs:
            raise TypeError('you give an unexpected keyword''argument "{0}"'.format(list(kwargs.keys())[0]))

    def get_input(self,test_data):
        self.test_data = test_data

    def get_neighbor(self):
        if self.distance == 'cosin':
            vector1 = self.train_data
            vector2 = self.test_data
            dominator = norm(vector1, axis=1).reshape((vector1.shape[0], 1)) * norm(vector2)
            dominator = np.where(dominator == 0,float('inf'),dominator)
            distance = np.dot(vector1, vector2).reshape((vector1.shape[0], 1)) / dominator
            min_index = distance.reshape((distance.shape[0],)).argsort()[0:self.k]
            self.neighbor_data,self.neighbor_label = self.train_data[min_index],self.train_label[min_index]

        elif self.distance == 'euclidean':
            vector1 = self.train_data  #(100L,3)
            vector2 = self.test_data  # (3L,)
            distance = np.sqrt(((vector1 - vector2) ** 2).sum(1))
            min_index = distance.reshape((distance.shape[0],)).argsort()[0:self.k]
            self.neighbor_data, self.neighbor_label = self.train_data[min_index], self.train_label[min_index]

    def get_label(self):
        self.get_neighbor()
        self.test_label = Counter(self.neighbor_label.reshape((self.neighbor_label.shape[0],))).most_common(1)[0][0]
        return self.test_label

if __name__ == '__main__':

    raw_data = np.loadtxt("./data/knnTestData.txt")
    np.save('./data/data.npy',raw_data)

    fulldata = np.load('./data/data.npy')
    data, label = np.split(fulldata, (3,), axis=1)
    train_data, test_data, train_label, test_label = train_test_split(data, label, random_state=1, train_size=0.2)

    precise = []
    for k in range(10, 200, 5):
        knn = KNN(k=k, train_data=train_data, train_label=train_label)
        erro = 0
        num = test_data.shape[0]
        for ite in range(0,num):
            knn.get_input(test_data[ite])
            guess = knn.get_label()
            if test_label[ite][0]!= guess:
                #print erro," actual label: ",test_label[ite][0]," || guess label: ",guess
                erro = erro + 1
            #print ite, " actual label: ", test_label[ite][0], " || guess label: ", guess

        print("total tests:  ",num,"\n erro: ",erro)
        precise.append((k,(num-erro)*1.0/num*100))
        print("precise:\n",(num-erro)*1.0/num*100,"%")

    K = []
    P = []
    for it in precise:
        K.append(it[0])
        P.append(it[1])
        print('K= ', it[0]," precise: ",it[1])

    plt.figure()
    plt.plot(K, P)
    plt.show()

