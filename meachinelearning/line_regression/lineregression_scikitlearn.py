
import numpy as np
from  sklearn import  linear_model
from sklearn.preprocessing import StandardScaler # 归一化处理

def linearRegression():
	print("加载数据....")
	data = loadtxtAndcsv_data('data.txt',',',np.float64)
	X = np.array(data[:,0:-1],dtype=np.float64)
	y = np.array(data[:,-1],dtype=np.float64)

	# 归一化操作
	scaler = StandardScaler()
	scaler.fit(X)
	x_train = scaler.transform(X)
	x_test = scaler.transform(np.array([1650,3]))

	#线性模型拟合
	model = linear_model.LinearRegression()
	model.fit(X_train,y)

	# 预测结果
	result = model.predict(x_test)
	print(model.coef_) #系数
	print(model.intercept_) #截距
	print(result) # 预测结果

# 加载text 和csv 文件
def loadtxtAndcsv_data(filename,split,dataType):
	return np.loadtxt(filename,delimiter=split,dtype=dataType)

# 加载npy 文件
def loadnpy_data(filenam):
	return np.load(filename)


if __name__ == '__main__':
	linearRegression()

	



