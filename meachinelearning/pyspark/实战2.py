# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:30:48 2018

@author: Administrator
"""

from pyspark import SparkContext,SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)
from pyspark.sql import Row

I =  [('Ankit',25),('Jalfaizy',22),('saurabh',20),('Bala',26)]
rdd = sc.parallelize(I)
people= rdd.map(lambda x:Row(name=x[0],age=x[1]))
schemaPeople = sqlContext.createDataFrame(people)
type(schemaPeople)

train = sqlContext.load(source="com.databricks.spark.csv", path = 'PATH/train.csv', header = True,inferSchema = True)
test = sqlContext.load(source="com.databricks.spark.csv", path = 'PATH/test-comb.csv', header = True,inferSchema = True)







