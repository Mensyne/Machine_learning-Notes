# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:08:27 2019

@author: admin
"""

## xgboost 可视化输出
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from xgboost.sklearn import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

import os
os.environ['PATH'] += os.pathsep+ 'D:/Graphviz2.38/bin'

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X  = DataFrame(X)
y = DataFrame(y)

X.columns = breast_cancer.feature_names


X.columns = ['l1','l2','l3','l4','l5','l6','l7','l8','l9','l10','l11','l12','l13','l14',
             'l15','l16','l17','l18','l19','l20','l21','l22','l23','l24','l25',
             'l26','l27','l28','l29','l30',]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = XGBClassifier(
    n_estimators=30,#三十棵树
    learning_rate =0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)
model_sklearn=clf.fit(X_train, y_train)
y_sklearn= clf.predict_proba(X_test)[:,1]

def ceate_feature_map(features):
    with open('xgb.fmap','w') as f:
        i = 0
        for feat in features:
            f.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
'''
X_train.columns在第一段代码中也已经设置过了。
特别需要注意：列名字中不能有空格。
'''
ceate_feature_map(X_train.columns)


plot_tree(clf, num_trees=0, fmap='xgb.fmap')
fig = plt.gcf()
fig.set_size_inches(30,30)
#plt.show()
fig.savefig('tree.png')









