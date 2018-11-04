import numpy as np

def load_data(inputfile):
	with open(inputfile) as f:
		feature_data = []
		label_data = []
		for line in f.readlines():
			feature_tmp = []
			feature_tmp.append(1)
			lines = line.strip().split('\t')
			for i in range(len(lines)-1):
				feature_tmp.append(float(lines[i]))
			label_data.append(lines[-1])
			feature_data.append(feature_tmp)
	return np.mat(feature_data),np.mat(label_data).T,len(set(label_data))

def cost(err,label_data):
	'''计算损失函数'''
	m = np.shape(err)[0]
	sum_cost = 0.0
	for i in range(m):
		if err[i,label_data[i,0]]/np.sum(err[i,:])>0:
			sum_cost -= np.log(err[i,label_data[i,0]]/np.sum(err[i,:]))
		else:
			sum_cost -=0
	return sum_cost/m

def gradientAscent(feature_data,label_data,k,maxCycle,alpha):
	''' 
	注意这个里面k(int)表示的是类别的个数为int类型
	alpha:表示学习率
	'''
	m,n = np.shape(feature_data)
	weights = np.mat(np.ones(n,k))  # 权重的初始化
	i= 0 
	while i<= maxCycle:
		err = np.exp(feature_data*weights)
		if i % 500 ==0:
			print("\t--iter:",i,", cost:",cost(err,label_data)
		rowsum =-err.sum(axis =1)
		rowsum = rowsum.repeat(k,axis =1)
		err = err/rowsum
		for x in range(m):
			err[x,label_data[x,0]] +=1
		weights = weights +(alpah/m)*feature_data.T*err
	return weights

def save_model(file_name,weights):
	''' weights(mat):softmax模型'''
		m,n = np.shape(weights)
		for i in range(m):
			w_tmp = []
			for j in range(n):
				w_tmp.append(str(weights[i,j]))
	with open(file_name,'w') as f_w:
		f_w.write("\t".join(w_tmp)+"\n")

if __name__ == '__main__':
	inputfile = "softInput.txt"
	print('-----------1.load data----------')
	feature,label,k = load_data(inputfile)
	print('-----------2.training-----------')
	weights = gradientAscent(feature,label,k,10000,0.4)
	print('-----------3.save model----------')
	save_model('weights',weights)

