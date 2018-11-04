# 使用sklearn 库
form sklearn.linear_model import LogisticRegression
import numpy as np
import random

def sigmoid(x):
	return 1.0 / (1+np.exp(-x))

# 改进的随机梯度上升算法
# dataMatrix - 数据数组 
#  classLabels - 数据标签
# numIter - 迭代次数
def stockGradAscent1(dataMatrix,classLabels,numIter= 150):
	m,n = np.shape(dataMatrix)
	weights = np.ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+i+j)+0.01
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex]-h
			weights = weights +alpah*error*dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights

# 随机梯度上升算法

'''
Parameters---dataMatIn 数据集  classLabels 数据标签
Returns weights.getA() - 求得权重数组(最优参数)
'''
def gradAscent(dataMatIn,classLabels):
	dataMatrix = np.mat(dataMatIn)]
	labelMat = np.mat(classLabels).transpose()
	m,n = np.shape(dataMatrix)
	alpha = 0.01
	maxCycles = 500
	weights = np.ones(n,1)
	for k in range(maxCycles):
		h = sigmoid(dataMatrix *weights)
		error = labelMat -h
		weights = weights +alpha*dataMatrix.transpose()*error
	return weights.getA()

# 使用logisitic 做预测

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingset = [];
	trainingLabels =[];
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr  =[]
		for i in range(len(currLine-1)):
			lineArr.append(float(currLine[i]))
		trainingset.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	trainWeights = stockGradAscent1(np.array(trainingset),trainingLabels,500)
	errCount = 0.0
	numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec+=1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[1]-1):
				errorCount +=1
	errorRate = (float(errorCount)/numTestVec)*100
	print("测试集错误率为：%.2f%%"%errorRate)


# 分类函数
def classifyVector(x,weights):
	prob = sigmoid(sum(x*weights))
	if prob >0.5:
		return 1.0
	else:
		return 0

"--------------------------------------------"
# 使用sklearn 构建logistic回归分类器
def colickSklearn():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr  =[]
		for i in range(len(currLine-1)):
			lineArr.append(float(currLine[i]))
		trainingset.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr  =[]
		for i in range(len(currLine-1)):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(trainingset,trainingLabels)
	test_accurcy = classifier.score(testSet,testLabels)*100
	print('正确率:%f%f'%test_accurcy)

if __name__ == '__main__':
	colickSklearn()

