"""
数据背景：众包任务价格制定中，地区任务的密度反映任务的密集程度、会员密度反映会员的密集程度。
而任务对会员的密度则可以用于刻画任务点周围会员的密集程度，从而体现任务被完成的相对概率。
此时训练样本为会员密度，测试样本为任务密度。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def localoutlierfactor(data, predict, k):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
    clf.fit(data)
    # 记录 k 邻域距离
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor'] = -clf._decision_function(predict.iloc[:, :-1])
    return predict


def plot_lof(result, method):

    plt.figure(figsize=(8, 4).add_subplot(111))
    plt.scatter(result[result['local outlier factor'] > method].index,
                result[result['local outfiler factor'] > method]['local outfiler factor'], c='red', s=50,
                marker='.', alpha=None, label="离群点")
    plt.scatter(result[result['local outlier factor'] <= method].index,
            result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
            marker='.', alpha=None, label='正常点')
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF 局部离群点检测',fontsize = 13)
    plt.ylabel("局部离群因子",fontsize = 15)
    plt.legend()
    plt.show()


def lof(data, predict=None, k=5, method=1, plot=False):
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass

    # 计算LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, method)
    # 根据阈值划分离群点 与正常点
    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outfiler factor')
    inliers = predict[predict['local outfiler factor'] <= method].sort_values(by="local outlier factor")
    return outliers, inliers


if __name__ == '__main__':
    # 获取任务对会员密度，取第5邻域，阈值为2（LOF大于2认为是离群值）
    # 设置文件的位置
    posi1 = pd.read_excel("./data/已结束项目任务数据.xls")
    lon1 = np.array(posi1['任务gps经度'][:])
    lat1 = np.array(posi1['任务gps维度'][:])
    A = list(zip(lat1, lon1))
    posi2 = pd.read_excel("./data/会员信息数据.xlsx")
    lon2 = np.array(posi2['任务gps经度'][:])
    lat2 = np.array(posi2['任务gps维度'][:])
    B = list(zip(lat2, lon2))
    outliers2, inliers2 = lof(B, A, k=5, method=2)
    for k, v in ([1, 5], [5, 2]):
        plt.figure('k=%d' % k)
        outliers2, inliers2 = lof(B, A, k=k, method=v)
        plt.scatter(np.array(A)[:, 0], np.array(A)[:, 1], s=10, c='b', alpha=0.5)
        plt.scatter(np.array(B)[:, 0], np.array(B)[:, 1], s=10, c='green', alpha=0.3)
        plt.scatter(outliers2[0], outliers2[1], s=10 + outliers2['local outlier factor'] * 100, c='r', alpha=0.2)
        plt.title('k = %d, method = %g' % (k, v))
