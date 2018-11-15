#!/usr/bin/env python
# -*- coding: utf-8 -*-
__title__ = ''
__author__ = 'tongshai'
__mtime__ = '2018/11/15'


from pyspark import SparkContext
import os
path = os.getcwd()
import re
sc = SparkContext(appName="test")
print(sc.version)

# 读取数据
data = sc.parallelize(
    [('Amber', 22), ('Alfred', 23), ('Skye',4), ('Albert', 12),
     ('Amber', 9)]).collect()
print(data)

# 选择某一个key 对应的value
print(data[1][1])

data_reduce = sc.parallelize([1, 2, .5, .1, 5, .2], 3)
print(data_reduce.reduce(lambda x,y:x/y))


data_key = sc.parallelize([('a', 4),('b', 3),('c', 2),('a', 8),('d', 2),('b', 1),('d', 3)],4)
print(data_key.reduceByKey(lambda x, y: x + y).collect())

print(data_reduce.count())

print(data_key.countByKey().items())

data_key.saveAsTextFile(path+"./data_key.txt")


def parseInput(row):

