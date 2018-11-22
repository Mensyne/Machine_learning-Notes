
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
import os
os.chdir('E:\\company_file\\learning\\Meachine_examples\\meachinelearning\\line_regression\\data')
def draw1(data, y):
    # 图画在一个画布中
    plt.plot(data['TV'], y, 'ro', label="TV")
    plt.plot(data["Radio"], y, 'g^', label='Radio')
    plt.plot(data["Newspaper"], y, 'mv', label='Newspaper')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def draw3(data, y):
    # 图画在三个画布中，figsize=(9, 9)表示画布大小为9X9
    plt.figure(figsize=(9, 9))

    # 311表示在9X9的画布分成3行1列的3X1个大小的第一块
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title("TV")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title("Radio")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title("Newspaper")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def linearRegession(x_train, y_train):
    # 创建线性回归模型
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)  # 线性回归模型训练数据集
    print('模型系数:%s' % model.coef_)     # 模型系数
    print('模型截距:%s' % model.intercept_) # 模型截距
    return linreg

def origin_regession(x, y):
    # 使用sklearn的train_test_split函数把数据切分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # 线性回归
    model = linearRegession(x_train, y_train)
    # 使用测试集数据测试数据
    y_pred = model.predict(np.array(x_test))
    mse = np.average((y_pred - np.array(y_test)) ** 2) # 均方误差值
    print('均方误差值：%s; %s' % (mse, np.sqrt(mse)))
    # 把实际结果和预测结果同时输出到图画中进行直观比较
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_pred, 'g-', linewidth=2, label='Pred')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
## LassoRegression 
def lassoRegession(x, y):
    # L1回归模型，即在线性模型的基础之增加绝对值和损失
    model = Lasso()
    alpha = np.logspace(-3, 2, 10)
    # 采用5折交叉验证
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha}, cv = 5)
    lasso_model.fit(x, y)
    print('best_params:%s' % lasso_model.best_params_)

    return lasso_model

def l1_regession(x, y):
    # 使用sklearn的train_test_split函数把数据切分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    # 线性回归
    #model = linearRegession(x_train, y_train)

    # L1即Lasso回归模型
    model = lassoRegession(x, y)

    # 使用测试集数据测试数据
    y_pred = model.predict(np.array(x_test))
    mse = np.average((y_pred - np.array(y_test)) ** 2) # 均方误差值
    print('均方误差值：%s; %s' % (mse, np.sqrt(mse)))

    # 把实际结果和预测结果同时输出到图画中进行直观比较
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_pred, 'g-', linewidth=2, label='Pred')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def ridgeRegession(x, y):
    # L2回归模型，即在线性模型的基础之增加平方和损失
    model = Ridge()
    alpha = np.logspace(-3, 2, 10)
    # 采用5折交叉验证
    ridge_model = GridSearchCV(model, param_grid={'alpha': alpha}, cv = 5)
    ridge_model.fit(x, y)
    print('best_params:%s' % ridge_model.best_params_)

    return ridge_model

def l2_regession(x, y):
    # 使用sklearn的train_test_split函数把数据切分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    # 线性回归
    #model = linearRegession(x_train, y_train)

    # L1即Lasso回归模型
    #model = lassoRegession(x, y)

    # L2即Ridge回归模型
    model = ridgeRegession(x, y)

    # 使用测试集数据测试数据
    y_pred = model.predict(np.array(x_test))
    mse = np.average((y_pred - np.array(y_test)) ** 2) # 均方误差值
    print('均方误差值：%s; %s' % (mse, np.sqrt(mse)))

    # 把实际结果和预测结果同时输出到图画中进行直观比较
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_pred, 'g-', linewidth=2, label='Pred')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    path = 'Advertising.csv'
    # pandas读取文件
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    draw3(x, y)
