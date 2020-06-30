#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:04:17 2020

@author: mensyne
"""

import os
os.chdir(r'/Users/mensyne/Desktop/多项式回归算法python 实现')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
## 拆分两个组件
X = data.iloc[:,1:2].values
y = data.iloc[:,2].values

## 拟合线性回归模型
from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(X,y)

## 将多项式回归模型拟合到两个分量X 和y上
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree =4)

X_poly = poly.fit_transform(X)

poly.fit(X_poly,y)

lin2 = LinearRegression()
lin2.fit(X_poly,y)


## 可视化回归结果
plt.scatter(X,y,color="blue")

plt.plot(X,lin.predict(X),color = "red")

plt.title("Linear Regression")
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()



plt.scatter(X, y, color = 'blue') 
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()


## 预测
lin.predict(110.0)

lin2.predict(poly.fit_transform(110.0))

