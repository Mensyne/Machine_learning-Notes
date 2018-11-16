#!/usr/bin/env python
# -*- coding: utf-8 -*-
__title__ = ''
__author__ = 'tongshai'
__mtime__ = '2018/11/16'

from pyspark.sql import  SparkSession

spark = SparkSession \
    .builder \
    .appName("test") \
    .getOrCreate()

sc = spark.sparkContext

df = spark.createDataFrame([
    (1, 144.5, 5.9, 33, 'M'),
    (2, 167.2, 5.4, 45, 'M'),
    (3, 124.1, 5.2, 23, 'F'),
    (4, 144.5, 5.9, 33, 'M'),
    (5, 133.2, 5.7, 54, 'F'),
    (3, 124.1, 5.2, 23, 'F'),
    (5, 129.2, 5.3, 42, 'M'),
    ],['id','weight','height','age','gender'])

print('Count of rows:{0}'.format(df.count()))
print('Count of distinct rows:{0}'.format(df.distinct().count()))

df = df.dropDuplicates()
df.show()

# 如果有多重复值 可以使用subset
df = df.dropDuplicates(subset = [c for c in df.columns if c !='id'])
df.show()


# 如果我们要计算总和和重复值 可以使用agg
import pyspark.sql.functions as fn

df.agg(
    fn.count('id').alias('count'),
    fn.countDistinct('id').alias('distinct')
).show()

# 设置唯一值
df.withColumn('new_id',fn.monotonically_increasing_id()).show()

# Missing 处理
df_miss = spark.createDataFrame([
        (1, 143.5, 5.6, 28,   'M',  100000),
        (2, 167.2, 5.4, 45,   'M',  None),
        (3, None , 5.2, None, None, None),
        (4, 144.5, 5.9, 33,   'M',  None),
        (5, 133.2, 5.7, 54,   'F',  None),
        (6, 124.1, 5.2, None, 'F',  None),
        (7, 129.2, 5.3, 42,   'M',  76000),
    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])

# 找到缺失值
df_miss.rdd.map(lambda row:(row['id'],sum([c==None for c in row]))).collect()

df_miss.where('id ==3').show()


