# coding:utf-8
# @Time :2018/11/28 10:49
# @Author: Mensyne
# @File :GDBT.py
# LR+GDBT

# 建树采用ensemble决策树
# 一棵树的区分性是具有一定的限制 但是多棵树可以获取多个具有区分度的特征组合 而且GBDT的每一棵树都会去学习前面的树的不足
# 建树算法为什么采用GBDT而不是RF
# 对于GDBT 而言 前面的树，特征分裂主要体现在对多数样本具有区分度的特征;后面的树 主要体现在经过前面n棵树 残差依然比较大的少数样本
# 优先选用在整体上具有区分度的特征再选用针对少数样本有区分度的特征
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier



#生成随机数据
np.random.seed(10)
X,Y = make_classification(n_samples=1000,n_features=30)
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,random_state=233,test_size=0.5)
X_train,X_train_lr,Y_train,Y_train_lr = train_test_split(X_train,Y_train,random_state=233,test_size=0.2)

# RandomForest+LogisticRegression
def RandomForestLR():
    RF = RandomForestClassifier(n_estimators=100,max_depth=4)
    RF.fit(X_train,Y_train)
    OHE = OneHotEncoder()
    OHE.fit(RF.apply(X_train))
    LR = LogisticRegression()
    LR.fit(OHE.transform(RF.apply(X_train_lr)),Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(RF.apply(X_test)))[:,1]
    fpr,tpr,_ = roc_curve(Y_test,Y_pred)
    auc = roc_auc_score(Y_test,Y_pred)
    print("RandomForest+LogisticRegression:",auc)
    return fpr,tpr

# Xgboost+LogisticRegression
def XGBoostLR():
    XGB = xgb.XGBClassifier(nthread=4,learning_rate=0.08,n_estimators =100,colsample_bytree =0.5)
    XGB.fit(X_train,Y_train)
    OHE = OneHotEncoder()
    OHE.fit(XGB.apply(X_train))
    LR = LogisticRegression(n_jobs=4,C=0.1,penalty="l1")
    LR.fit(OHE.transform(XGB.apply(X_train_lr)),Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(XGB.apply(X_test)))[:,1]
    fpr,tpr,_ = roc_curve(Y_test,Y_pred)
    auc = roc_auc_score(Y_test,Y_pred)
    print('XGBoost+logisticRegression:',auc)
    return fpr,tpr

# GradientBoosting+LogisticRegression
def GBDTLR():
    GBDT= GradientBoostingClassifier(n_estimators=10)
    GBDT.fit(X_train,Y_train)
    OHE = OneHotEncoder()
    OHE.fit(GBDT.apply(X_train)[:,:,0])
    LR = LogisticRegression()
    LR.fit(OHE.transform(GBDT.apply(X_train_lr)[:,:,0]),Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(GBDT.apply(X_test)[:,:,0]))[:,1]
    fpr,tpr,_ = roc_curve(Y_test,Y_pred)
    auc= roc_auc_score(Y_test,Y_pred)
    print('GradientBoosting+LogisticRegression:',auc)
    return fpr,tpr

# LogsicRegression
def LR():
    LR = LogisticRegression(n_jobs=4,C=0.1,penalty='l1')
    LR.fit(X_train,Y_train)
    Y_pred = LR.predict_proba(X_test)[:,1]
    fpr,tpr,_ = roc_curve(Y_test,Y_pred)
    auc = roc_auc_score(Y_test,Y_pred)
    print('LogisticRegression:',auc)
    return fpr,tpr

# XGBOOST

def XGBoost():
    XGB = xgb.XGBClassifier(nthread=4,learning_rate=0.08,n_estimators=100)
    XGB.fit(X_train,Y_train)
    Y_pred = XGB.predict_proba(X_test)[:,1]
    fpr,tpr,_ = roc_curve(Y_test,Y_pred)
    auc = roc_auc_score(Y_test,Y_pred)
    print('XGBoost:',auc)
    return fpr,tpr

if __name__ == '__main__':
    fpr_xgb_lr, tpr_xgb_lr = XGBoostLR()
    fpr_xgb, tpr_xgb = XGBoost()
    fpr_lr, tpr_lr = LR()
    fpr_rf_lr, tpr_rf_lr = RandomForestLR()
    fpr_gbdt_lr, tpr_gbdt_lr = GBDTLR()

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBT + LR')
    plt.plot(fpr_xgb, tpr_xgb, label='XGB')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBT + LR')
    plt.plot(fpr_xgb, tpr_xgb, label='XGB')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()
















