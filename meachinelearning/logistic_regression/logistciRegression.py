import numpy as np 

def load_data(file_name):
	'''导入训练数据'''
	f = open(file_name)
	feature_data = []
	label_data = []
	for line in f.readlines():
		feature_tmp = []
		lable_tmp = []
		lines = line.strip().split("\t")
		feature_tmp.append(1)  # 偏置顶
		for i in range(len(lines)-1):
			feature_tmp.append(float(lines[i]))
		lable_tmp.append(float(lines[-1]))

		feature_data.append(feature_tmp)
		label_data.append(label_tmp)
	f.close()  #关闭文件
	return np.mat(feature_data) ,np.mat(label_data)

def sig(x):
	return 1.0/(1+np.exp(-x))

def lr_train_bgd(feature,lable,maxCycle,alpha):
	n = np.shape(feature)[1]
	w = np.mat(np.ones((n,1)))
	i = 0
	while i <=maxCycle:
		i+=1
		h = sig(feature*w)
		err = label -h
		if i %100 ==0:
			print('\t------iter='+str(i)+",train error rate"+str(error_rate(h,label)))
		w = w + alpha*feature.T*err  # 权重修正
	return w


def error_rate(h,label):
	m = np.shape(h)[0]
	sum_err = 0.0
	for i in range(m):
		if h[i,0] > 0  and (1-h[i,0]) >0:
			sum_err -= (label[i,0]*np.log(h[i,0]) +\
				(1-label[i,0])*np.log(1-h[i,0]))
		else:
			sum_err -=0
	return sum_err/m

def save_model(file_name,w):
	m = np.shape(w)[0]
	f_w = open(file_name,'w')
	w_array = []
	for i in range(m):
		w_array.append(str(w[i,0]))
	f_w.write("\t".join(w_array))
	f_w.close()

if __name__ == '__main__':
	print("---------1. load data---------")
	feature ,label = load_data('data.txt')
	print("----------2. training ---------")
	w = Ir_train_bgd(feature,label,1000,0.01)
	print('----------3. save model -------')
	save_model("weight",w)
