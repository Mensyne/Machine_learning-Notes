# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:54:15 2018

@author: tongshai
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge




class Regression:
    def __init__(self, x, y, method, plot, test_size=0.7,start=0,stop=1,iteration=50,cv=5):
        self.x = x
        self.y = y
        self.method = method
        self.plot = plot
        self.test_size = test_size
        self.start=start
        self.stop = stop
        self.iteration = iteration
        self.cv= cv
        self.alpha = np.logspace(self.start, self.stop, self.iteration)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y,
                                                                                test_size=self.test_size)

    def draw(self):
        if self.plot == "1":
            # 图画在一个画布中
            plt.plot(data['TV'], y, 'ro', label="TV")
            plt.plot(data["radio"], y, 'g^', label='radio')
            plt.plot(data["newspaper"], y, 'mv', label='newspaper')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()
        elif self.plot ==  "3":

            # 图画在三个画布中，figsize=(9, 9)表示画布大小为9X9
            plt.figure(figsize=(9, 9))
            # 311表示在9X9的画布分成3行1列的3X1个大小的第一块
            plt.subplot(311)
            plt.plot(data['TV'], y, 'ro')
            plt.title("TV")
            plt.grid(True)

            plt.subplot(312)
            plt.plot(data['radio'], y, 'g^')
            plt.title("radio")
            plt.grid(True)

            plt.subplot(313)
            plt.plot(data['newspaper'], y, 'b*')
            plt.title("newspaper")
            plt.grid(True)

            plt.tight_layout()
            plt.show()


    def regression(self):
        if self.method == "origin":
            # 线性回归
            self.regession_model = LinearRegression(self.train_x,self.train_y)
            print('模型系数:%s' % self.regession_model.coef_)  # 模型系数
            print('模型截距:%s' % self.regession_model.intercept_)  # 模型截距
            # 使用测试集数据测试数据
            self.pred_y = self.regession_model.predict(np.array(self.test_x))
            self.mse = np.average((self.pred_y - np.array(self.test_y) **2))  # 均方误差值
            print('均方误差值：%s; %s' % (self.mse, np.sqrt(self.mse)))


        ## LassoRegression
        elif self.method == "l1":
            self.model = Lasso()
            # 采用5折交叉验证
            self.lasso_model = GridSearchCV(self.model,param_grid={'alpha':self.alpha},cv=self.cv)
            self.lasso_model.fit(self.train_x,self.train_y)
            print("best_params:%s"%self.lasso_model.best_params_)
            # 使用测试集数据测试数据
            self.pred_y = self.lasso_model.predict(np.array(self.test_x))
            self.mse = np.average((self.pred_y - np.array(self.test_y) ** 2))  # 均方误差值
            print('均方误差值：%s; %s' % (self.mse, np.sqrt(self.mse)))

        ## RidgeRegression
        elif self.method == "l2":
            self.model = Ridge()
            self.ridge_model = GridSearchCV(self.model,param_grid={'alpha':self.alpha},cv = self.cv)
            self.ridge_model.fit(self.train_x,self.train_y)
            print("best_params:%s"%self.ridge_model.best_params_)
            self.pred_y = self.ridge_model.predict(np.array(self.test_x))
            self.mse = np.average((self.pred_y- np.array(self.test_y)) ** 2)  # 均方误差值
            print('均方误差值：%s; %s' % (self.mse, np.sqrt(self.mse)))
        # 把实际结果和预测结果同时输出到图画中进行直观比较
        t = np.arange(len(self.test_x))
        plt.plot(t, self.test_y, 'r-', linewidth=2, label='Test')
        plt.plot(t, self.pred_y, 'g-', linewidth=2, label='label')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()




if __name__ == '__main__':
    path = './data/Advertising.csv'
    # pandas读取文件
    data = pd.read_csv(path)
    x = data[['TV', 'radio', 'newspaper']]
    y = data['sales']
    model = Regression(x,y,"l2","3",start=-3,stop=1)
    model.draw()
    model.regression()


