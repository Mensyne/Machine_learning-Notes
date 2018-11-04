import numpy as np
from  random import normalvariate  # 正态分布

def loadDataSet(data):
	dataMat = []
	labelMat = []
	fr = open(data)  # 打开文件
	for line in fr.readlines():
		lines = line.strip().split("\t")
		lineArr = []
		for i in range(len(lines)-1):
			lineArr.append(float(lines[i]))
		dataMat.append(lineArr)

		labelMat.append(float(lines[-1]*2-1))  # 转换成（-1,1）
	fr.close()
	return dataMat,labelMat

def sigmoid(x):
	return 1.0 / (1+np.exp(-x))


def initialize_v(n,k):
	'''初始化交叉项'''
	v = np.mat(np.zeros((n,k)))
	for i in range(n):
		for j in range(k):
			# 利用正态分布生成每一个权重
			v[i,j] = normalvariate(0,0.2)
	return v

def stocGradAscent(dataMatrix,classlabels,k,max_iter,alpha):
	'''利用随机梯度下降法训练FM模型'''
	m,n = np.shape(dataMatrix)
	# 初始化参数
	w = np.zeros((n,1))  # 其中n是特征的个数
	w0 = 0  # 偏置项
	v = initialize_v(n,k)  # 初始化v
	# 训练
	for it in range(max_iter):
		for x in range(m):    # 随机优化 对每一个样本而言
			inter_1 = dataMatrix[x]*v
			inter_2 = np.multiply(dataMatrix[x],dataMatrix[x])*np.multiply(v,v) # multiply 对应的元素相乘
			# 完成交叉项
			interaction = np.sum(np.multiply(inter_1,inter_1)-inter_2) /2
			p = w0+ dataMatrix[x]*w+interaction  # 计算预测的输出
			loss  = sigmoid(classlabels[x]*p[0,0]) -1
			w0 = w0 -alpha*loss*classLabels[x]
			for i in range(n):
				if dataMatrix[x,i] !=0:
					w[i,0] = w[i,0] -alpha*loss*classLabels[x]*dataMatrix[x,i]
					for j in range(k):
						v[i,j] = v[i,j] -alpha*loss*classLabels[x]*(dataMatrix[x,i]*inter_1[0,j]-v[i,j]*dataMatrix[x,i]*dataMatrix[x,i])
		# 计算损失函数的值
		if it %1000 ==0:
			print("\t------iter:"+it+",cost:"+getCost(getPrediction(np.mat(dataMatrix),w0,w,v),classlabels))				
	return w0,w,v

def getCost(predict,classLabels):
	m = len(predict)
	error = 0.0
	for i in range(m):
		error -= np.log(sigmoid(predict[i]*classlabels[i]))
	return error

def getPrediction(dataMatrix,w0,w,v):
	m = np.shape(dataMatrix)[0]
	result = []
	for x in range(m):
		inter_1 = dataMatrix[x]*v
		inter_2 = np.multiply(dataMatrix[x],dataMatrix[x])*np.multiply(v,v)
		interaction = np.sum(np.multiply(inter_1,inter_1) - inter_2)/2
		p = w0 + dataMatrix[x]*w+interaction  # 计算预测的输出
		pre = sigmoid(p[0,0])
		result.append(pre)
	return result


def getAccuracy(predict,classlabels):
	m = len(predict)
	allItem = 0
	error = 0
	for i in range(x):
		allItem +=1
		if float(predict[i]) <0.5 and classlabels[i] == 1.0:
			error +=1
		elif float(predict[i]) >= 0.5 and classlabels[i] == -1.0:
			error +=1
		else:
			continue
	return float(error) / allItem

def save_model(file_name,w0,w,v):
	with open(file_name,'w') as f:
		f.write(str(w0+"\n"))
		# 保存在一次项的权重
		w_array = []
		m = np.shape(w)[0]
		for i in range(m):
			w_array.append(str(w[i,0]))
		f.write("\t".join(w_array) + "\n")
		# 保存交叉项数量
		m1,n1 = np.shape(v)
		for i in range(m1):
			v_tmp = []
			for j in range(n1):
				v_tmp.append(str(v[i,j]))
			f.write("\t".join(v_tmp)+"\n")

if __name__ == '__main__':
	print("--------1.load data-------------")
	dataTrain,labelTrain = loadDataSet("data_1.txt")
	print("---------2.learning--------------")
	w0,x,v = stocGradAscent(np.mat(dataTrain),labelTrain,3,10000,0.01)
	predict_result = getPrediction(np.mat(dataTrain),w0,w,v)
	print("----------3.training accuracy；%f"%(1-getAccuracy(predict_result,labelTrain)))
	save_model("weights",w0,w,v)











