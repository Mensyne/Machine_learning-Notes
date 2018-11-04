import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X1,y1 = datasets.make_circles(n_samples=5000,
                              factor=.6,noise=.05)

X2,y2 = datasets.make_blobs(n_samples=1000,
                            n_features=2,centers=[[1.2,1.2]],
                            cluster_std=[[.1]],random_state=9)
X=np.concatenate((X1,X2))
plt.scatter(X[:,0],X[:,1],marker='o')
plt.show()

# 使用的K-means来聚类
# from sklearn.cluster import KMeans
# y_pred = KMeans(n_clusters=3,random_state=9).fit_predict(X)
# plt.scatter(X[:,0],X[:,1],c = y_pred)
# plt.show()

# 使用DBSCAN
from sklearn.cluster import DBSCAN
y_pred =  DBSCAN().fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()


