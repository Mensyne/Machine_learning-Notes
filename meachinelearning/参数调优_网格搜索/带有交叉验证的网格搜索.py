#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:20:16 2020

@author: mensyne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import StratifiedKFold,GridSearchCV,train_test_split

from xgboost import XGBClassifier


pima = pd.read_csv(r'/Users/mensyne/Downloads/meachine-learning-note/meachinelearning/参数调优_网格搜索/pima-indians-diabetes.csv')

x = pima.iloc[:,0:8]
y = pima.iloc[:,8]

seed = 7
test_size = 0.3
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size= test_size,random_state = seed)
model = XGBClassifier()
learning_rate =  [0.0001,0.001,0.01,0.1,0.2,0.3]
gamma = [1,0.1,0.01,0.0001]

param_grid = dict(learning_rate = learning_rate,gamma = gamma)
kflod = StratifiedKFold(n_splits = 10,shuffle = True,random_state = 7)
grid_search = GridSearchCV(model,param_grid,scoring = "roc_auc",n_jobs = -1,cv = kflod)
grid_result = grid_search.fit(x_train,y_train)


column = list(param_grid.keys())
param_df= pd.DataFrame(columns = ['iteration'] + column + ['auc'])
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

for idx,row_param in enumerate(params):
    row = [idx]
    for k in column:
        row.append(row_param[k])
    row.append(means[idx])
    param_df.loc[idx]  = row


print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
