
import numpy as np
import panada as pd
import os
import matplotlib.pyplot as plt

path = os.getcwd()
# 设置文件的位置
posi = pd.read_excel(path+"./data/已结束项目任务数据.xls")
lon = np.array(posi['任务gps经度'][:])
lat = np.array(posi['任务gps维度'][:])
A = list(zip(lat,lon))

def plot_lof(result,method):
	plt.rcParams['font.snans-serif'] =['SimHei'] # 显示中文标签
	plt.rcParams['axes.unicode_minus'] = False  # 显示正常负号
	plt.figure(figsize=(8,4).add_subplot(111))
	plt.scatter(result[result['local outlier factor'] > method].index,
		result[result['local outfiler factor'] > method]['local outfiler factor'],c='red',s=50,
		marker='.',alpha=None,label = "离群点")
    plt.scatter(result[result['local outlier factor'] <= method].index,
            result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
            marker='.', alpha=None, label='正常点')
    plt.hlines(method,-2,2+max(result.index),linestyles ='--')
    plt.xlim(-2,2+max(result.index))
    plt.title('LOF 局部离群点检测'，fontsize=13)
    plt.ylabel("局部离群因子"，fontsize=15)
    plt.legend()
    plt.show()

def lof(data,predict=None,k=5,method=1,plot=False):
	try:
		if predict == None:
			predict = data.copy()
	except Exception:
		pass

	# 计算LOF 离群因子
	predict = localoutlierfactor(data,predict,k)
	if plot == True:
		plot_lof(predict,method)
	# 根据阈值划分离群点 与正常点
	outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outfiler factor')
	inliers = predict[predict['local outfiler factor'] <= method].sort_values(by="local outlier factor")
	return outfiler,inliers

if __name__ == '__main__':
# 获取任务密度 取第5 邻域阈值为2
	outliers1,inliers1 = lof(A,k,method=2)
	for k in [3,5,10]:
		    plt.figure('k=%d'%k)
    		outliers1, inliers1 = lof(A, k=k, method = 2)
   			plt.scatter(np.array(A)[:,0],np.array(A)[:,1],s = 10,c='b',alpha = 0.5)
    		plt.scatter(outliers1[0],outliers1[1],s = 10+outliers1['local outlier factor']*100,c='r',alpha = 0.2)
    		plt.title('k=%d' % k)


