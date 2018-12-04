# coding:utf-8
# @Time :2018/12/4 9:18
# @Author: Mensyne
# @File :实战案例.py

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("test").getOrCreate()
sc = spark.sparkContext

#读取数据
data = spark.read.csv(r'./data/train.csv',header=True,sep=',')

# 去除一些不需要的列名
drop_list = ['Dates','DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
data = data.select([column for column in data.columns if column not in drop_list])
print(data.show(5))
print(data.printSchema())

# 相关的聚合操作
from pyspark.sql.functions import col
data.groupBy('Category').count().orderBy(col("count").desc()).show()

data.groupBy('Descript').count().orderBy(col('count').desc()).show()

from pyspark.ml.feature import RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.classification import LogisticRegression

regexTokenizer = RegexTokenizer(inputCol='Descript',outputCol='words',pattern="\\w")
add_stopwords = ["http","https","amp","rt","t","c","the"]
stopwordsRemover = StopWordsRemover(inputCol ='words',outputCol = 'filtered').setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered",outputCol="features",vocabSize=10000,minDF=5)

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder,StringIndexer,VectorAssembler
label_stringIdx = StringIndexer(inputCol ="Category",outputCol="label")
pipeline = Pipeline(stages=[regexTokenizer,stopwordsRemover,countVectors,label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)

## 测试和训练数据划分
(trainingData,testData) = dataset.randomSplit([0.7,0.3],seed=100)
print("Training Dataset Count:"+str(trainingData.count()))
print("Test DataSet Count:"+str(testData.count()))

## 以词频作为特征 利用LR 进行分类
lr = LogisticRegression(maxIter=20,regParam=0.3,elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0).\
    select("Descript","Category","probability","label","prediction")\
    .orderBy("probability", ascending=False)\
    .show(n = 10, truncate = 30)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)

#TF-IDF作为特征
from pyspark.ml.feature import HashingTF, IDF
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
#minDocFreq: remove sparse terms
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf,
label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset1 = pipelineFit.transform(data)
(trainingData1, testData1) = dataset1.randomSplit([0.7, 0.3], seed = 100)
lr1 = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel1 = lr1.fit(trainingData)
predictions1 = lrModel1.transform(testData1)
predictions.filter(predictions1['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

# 交叉验证

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset2 = pipelineFit.transform(data)
(trainingData2, testData2) = dataset1.randomSplit([0.7, 0.3], seed = 100)
lr2 = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr2.regParam, [0.1, 0.3, 0.5]) # regularization parameter
             .addGrid(lr2.elasticNetParam, [0.0, 0.1, 0.2])
                  # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10, 20, 50]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr2, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(trainingData2)

predictions = cvModel.transform(testData2)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)


## 朴素贝叶斯
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)
predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

## 随机森林
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)
# Train model with Training Data
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

