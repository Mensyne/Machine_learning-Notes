#!/usr/bin/env python
# -*- coding: utf-8 -*-
__title__ = ''
__author__ = 'tongshai'
__mtime__ = '2018/11/12'

import numpy as np


class DecisionTree():
    """"
    决策树使用方法

    """

    def __init__(self,mode='C4.5'):
        self._tree = None
        if mode == 'C4.5' or mode == 'ID3':
            self._mode = mode
        else:
            raise Exception('mode should be C4.5 or ID3')


    def _calcEntorpy(self,y):
        """函数功能：计算熵
        """
        num = y.shape[0]
        # 使用字典labelCounts 来储存
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1

            # 计算熵
            entropy = 0.0

            for key in labelCounts:
                prob = float(labelCounts[key])/num
                entropy -= prob*np.log2(prob)
        return entropy

    def _splitDataSet(self,X,y,index,value):
        """返回数据集中特征下标为index 特征值等于value 的子数据集"""
        ret = []
        featVec = X[:,index]
        X = X[:[i for i in range(X.shape[1]) if i != index]]
        for i in range(len(featVec)):
            if featVec[i] == value:
                ret.append(i)
        return X[ret,:],y[ret]

    def _chooseBestFeaturesToSplit_ID3(self,X,y):
        """

        :param X:
        :param y:
        :return:
        """







