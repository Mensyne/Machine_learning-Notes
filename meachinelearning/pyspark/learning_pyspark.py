#!/usr/bin/env python
# -*- coding: utf-8 -*-
__title__ = ''
__author__ = 'tongshai'
__mtime__ = '2018/11/15'


from pyspark import SparkContext
from pyspark.sql import  SparkSession
import os
filepath = os.getcwd()
spark = SparkSession \
    .builder \
    .appName("test") \
    .getOrCreate()

sc = spark.sparkContext
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

# .foreach(...)
# A method that applies the same function to each element of the RDD in an iterative way
def f(x):
    print(x)

data_key.foreach(f)



stringJSONRDD = sc.parallelize((""" 
  { "id": "123",
    "name": "Katie",
    "age": 19,
    "eyeColor": "brown"
  }""",
   """{
    "id": "234",
    "name": "Michael",
    "age": 22,
    "eyeColor": "green"
  }""",
  """{
    "id": "345",
    "name": "Simone",
    "age": 23,
    "eyeColor": "blue"
  }""")
)

swimmersJSON = spark.read.json(stringJSONRDD)
swimmersJSON.createOrReplaceTempView("swimmersJSON")
swimmersJSON.show()

# Sql Query
spark.sql("select * from swimmersJSON").collect()

swimmersJSON.printSchema()


from pyspark.sql.types import *

stringCSVRDD = sc.parallelize([(123, 'Katie', 19, 'brown'), (234, 'Michael', 22, 'green'), (345, 'Simone', 23, 'blue')])
schemaString ="id name age eyeColor"
schema =StructType([
    StructField("id",LongType(),True),
    StructField("name",StringType(),True),
    StructField("age",LongType(),True),
    StructField("eyeColor",StringType(),True)
    ])
# Apply the schema to the RDD and Create DataFrame
swimmers = spark.createDataFrame(stringCSVRDD,schema)
swimmers.createOrReplaceTempView("swimmers")
swimmers.printSchema()
# Excute SQL Query and return the data
spark.sql("select * from swimmers").show()
spark.sql("select count(1) from swimmers").show()

swimmers.select("id","age").filter("age=22").show()
swimmers.select(swimmers.id,swimmers.age).filter(swimmers.age==22).show()

spark.sql("select name,eyeColor from swimmers where eyeColor like 'b%'").show()

# DataFrame API
swimmers.show()
# Using Databricks 'display' command to view the data easier

# get count of rows
swimmers.count()

#Get the id ,ahe where age =22
swimmers.select('id','age').filter('age=22').show()

swimmers.select("name","eyeColor").filter("eyeColor like '%b%'").show()

# Set File Paths
flightPerfFilePath = filepath+"\\flight-data\\departuredelays.csv"
airportsFilePath = filepath +"\\flight-data\\airport-codes-na.txt"

airports = spark.read.csv(airportsFilePath,header='true',inferSchema='true',
                          sep ="\t")
airports.createOrReplaceTempView("airports")

flightPerf = spark.read.csv(flightPerfFilePath,header='true')
flightPerf.createOrReplaceTempView("FlightPerformance")
#Cache the Departure Delays dataset
flightPerf.cache()
spark.sql("select a.City, f.origin, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.State = 'WA' group by a.City, f.origin order by sum(f.delay) desc").show()


spark.sql("select a.State, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.Country = 'USA' group by a.State ").show()








