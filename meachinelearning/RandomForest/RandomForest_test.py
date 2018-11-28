# coding:utf-8
# @Time :2018/11/28 23:47
# @Author: Mensyne
# @File :RandomForest_test.py


import numpy as np
import math
from collections import Counter, defaultdict

class node:
    def __init__(self, fea=-1, val=None, res=None, right=None, left=None):
        self.fea = fea  # 特征
        self.val = val  # 特征对应的值
        self.res = res  # 叶节点标记
        self.right = right
        self.left = left

class CART_CLF:
    def __init__(self, epsilon=1e-3, min_sample=1):
        self.epsilon = epsilon
        self.min_sample = min_sample  # 叶节点含有的最少样本数
        self.tree = None

    def validate(self,X_data,y_data, ratio=0.15):
        N = X_data.shape[0]
        size = int(N * ratio)
        inds = np.random.permutation(range(N))
        for i in range(int(N / size)):
            test_ind = inds[i * size:(i + 1) * size]
            train_ind = list(set(range(N)) - set(test_ind))
            yield X_data[train_ind], y_data[train_ind], X_data[test_ind], y_data[test_ind]

    def getGini(self, y_data):
        # 计算基尼指数
        c = Counter(y_data)
        return 1 - sum([(val / y_data.shape[0]) ** 2 for val in c.values()])

    def getFeaGini(self, set1, set2):
        # 计算某个特征及相应的某个特征值组成的切分节点的基尼指数
        num = set1.shape[0] + set2.shape[0]
        return set1.shape[0] / num * self.getGini(set1) + set2.shape[0] / num * self.getGini(set2)

    def bestSplit(self, splits_set, X_data, y_data):
        # 返回所有切分点的基尼指数，以字典形式存储。键为split，是一个元组，第一个元素为最优切分特征，第二个为该特征对应的最优切分值
        pre_gini = self.getGini(y_data)
        subdata_inds = defaultdict(list)  # 切分点以及相应的样本点的索引
        for split in splits_set:
            for ind, sample in enumerate(X_data):
                if sample[split[0]] == split[1]:
                    subdata_inds[split].append(ind)
        min_gini = 1
        best_split = None
        best_set = None
        for split, data_ind in subdata_inds.items():
            set1 = y_data[data_ind]  # 满足切分点的条件，则为左子树
            set2_inds = list(set(range(y_data.shape[0])) - set(data_ind))
            set2 = y_data[set2_inds]
            if set1.shape[0] < 1 or set2.shape[0] < 1:
                continue
            now_gini = self.getFeaGini(set1, set2)
            if now_gini < min_gini:
                min_gini = now_gini
                best_split = split
                best_set = (data_ind, set2_inds)
        if abs(pre_gini - min_gini) < self.epsilon:  # 若切分后基尼指数下降未超过阈值则停止切分
            best_split = None
        return best_split, best_set, min_gini

    def buildTree(self, splits_set, X_data, y_data):
        if y_data.shape[0] < self.min_sample:  # 数据集小于阈值直接设为叶节点
            return node(res=Counter(y_data).most_common(1)[0][0])
        best_split, best_set, min_gini = self.bestSplit(splits_set, X_data, y_data)
        if best_split is None:  # 基尼指数下降小于阈值，则终止切分，设为叶节点
            return node(res=Counter(y_data).most_common(1)[0][0])
        else:
            splits_set.remove(best_split)
            left = self.buildTree(splits_set, X_data[best_set[0]], y_data[best_set[0]])
            right = self.buildTree(splits_set, X_data[best_set[1]], y_data[best_set[1]])
            return node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, X_data, y_data):
        # 训练模型，CART分类树与ID3最大的不同是，CART建立的是二叉树，每个节点是特征及其对应的某个值组成的元组
        # 特征可以多次使用
        splits_set = []
        for fea in range(X_data.shape[1]):
            unique_vals = np.unique(X_data[:, fea])
            if unique_vals.shape[0] < 2:
                continue
            elif unique_vals.shape[0] == 2:  # 若特征取值只有2个，则只有一个切分点，非此即彼
                splits_set.append((fea, unique_vals[0]))
            else:
                for val in unique_vals:
                    splits_set.append((fea, val))
        self.tree = self.buildTree(splits_set, X_data, y_data)
        return

    def predict(self, x):
        def helper(x, tree):
            if tree.res is not None:  # 表明到达叶节点
                return tree.res
            else:
                if x[tree.fea] == tree.val:  # "是" 返回左子树
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x, branch)

        return helper(x, self.tree)

    def disp_tree(self):
        # 打印树
        self.disp_helper(self.tree)
        return

    def disp_helper(self, current_node):
        # 前序遍历
        print(current_node.fea, current_node.val, current_node.res)
        if current_node.res is not None:
            return
        self.disp_helper(current_node.left)
        self.disp_helper(current_node.right)
        return


class RandomForest:
    def __init__(self, n_tree=6, n_fea=None, ri_rc=True, L=None, epsilon=1e-3, min_sample=1):
        self.n_tree = n_tree
        self.n_fea = n_fea  # 每棵树中特征的数量
        self.ri_rc = ri_rc  # 判定特征的选择选用RI还是RC, 特征比较少时使用RC
        self.L = L # 选择RC时，进行线性组合的特征个数
        self.tree_list = []  # 随机森林中子树的list

        self.epsilon = epsilon
        self.min_sample = min_sample  # 叶节点含有的最少样本数

        self.D = None  # 输入数据维度
        self.N = None

    def init_param(self, X_data):
        # 初始化参数
        self.D = X_data.shape[1]
        self.N = X_data.shape[0]
        if self.n_fea is None:
            self.n_fea = int(math.log2(self.D) + 1)  # 默认选择特征的个数
        return

    def validate(self,X_data, y_data, ratio=0.15):
        N = X_data.shape[0]
        size = int(N * ratio)
        inds = np.random.permutation(range(N))
        for i in range(int(N / size)):
            test_ind = inds[i * size:(i + 1) * size]
            train_ind = list(set(range(N)) - set(test_ind))
            yield X_data[train_ind], y_data[train_ind], X_data[test_ind], y_data[test_ind]

    def extract_fea(self):
        # 从原数据中抽取特征(RI)或线性组合构建新特征(RC)
        if self.ri_rc:
            if self.n_fea > self.D:
                raise ValueError('the number of features should be lower than dimention of data while RI is chosen')
            fea_arr = np.random.choice(self.D, self.n_fea, replace=False)
        else:
            fea_arr = np.zeros((self.n_fea, self.D))
            for i in range(self.n_fea):
                out_fea = np.random.choice(self.D, self.L, replace=False)
                coeff = np.random.uniform(-1, 1, self.D)  # [-1,1]上的均匀分布来产生每个特征前的系数
                coeff[out_fea] = 0
                fea_arr[i] = coeff
        return fea_arr

    def extract_data(self, X_data, y_data):
        # 从原数据中有放回的抽取样本，构成每个决策树的自助样本集
        fea_arr = self.extract_fea()  # col_index or coeffs
        inds = np.unique(np.random.choice(self.N, self.N))  # row_index, 有放回抽取样本
        sub_X = X_data[inds]
        sub_y = y_data[inds]
        if self.ri_rc:
            sub_X = sub_X[:, fea_arr]
        else:
            sub_X = sub_X @ fea_arr.T
        return sub_X, sub_y, fea_arr

    def fit(self, X_data, y_data):
        # 训练主函数
        self.init_param(X_data)
        for i in range(self.n_tree):
            sub_X, sub_y, fea_arr = self.extract_data(X_data, y_data)
            subtree = CART_CLF(epsilon=self.epsilon, min_sample=self.min_sample)
            subtree.fit(sub_X, sub_y)
            self.tree_list.append((subtree, fea_arr))  # 保存训练后的树及其选用的特征，以便后续预测时使用
        return

    def predict(self, X):
        # 预测，多数表决
        res = defaultdict(int)  # 存储每个类得到的票数
        for item in self.tree_list:
            subtree, fea_arr = item
            if self.ri_rc:
                X_modify = X[fea_arr]
            else:
                X_modify = (np.array([X]) @ fea_arr.T)[0]
            label = subtree.predict(X_modify)
            res[label] += 1
        return max(res, key=res.get)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    data = load_iris()
    X_data = data['data']
    y_data = data['target']
    randomforest =RandomForest()
    g = randomforest.validate(X_data, y_data, ratio=0.2)
    for item in g:
        X_train, y_train, X_test, y_test = item
        RF = RandomForest(n_tree=50, n_fea=2, ri_rc=True)
        RF.fit(X_train, y_train)
        score = 0
        for X, y in zip(X_test, y_test):
            if RF.predict(X) == y:
                score += 1
        print(score / len(y_test))