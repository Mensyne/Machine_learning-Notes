
import numpy as np
import math

class SVM:
    def __init__(self,epsilon =1e-5,maxstep=500,C=1.0,kernel_option=True,gamma = None):
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.C =C
        self.kernel_option = kernel_option
        self.gamma = gamma

        self.kernel_arr = None
        self.X =None
        self.y = None
        self.alpha_arr = None
        self.b= 0
        self.err_arr = None

        self.N = None

    def init_param(self,X_data,y_data):
        # 初始化参数
        self.N = X_data.shape[0]
        self.X = X_data
        self.y = y_data
        if self.gamma is None:
            self.gamma =1.0 /X_data.shape[1]
        self.cal_kernel(X_data)
        self.alpha_arr = np.zeros(self.N)
        self.err_arr = -self.y
        return
    def _gaussian_dot(self,x1,x2):
        # 计算两个样本之间的高斯内积
        return math.exp(-self.gamma*np.square(x1-x2).sum())

    def cal_kernel(self,X_data):
        # 计算核内积矩阵
        if self.kernel_option:
            self.kernel_arr = np.ones((self.N,self.N))
            for i in range(self.N):
                for j in range(i+1,self.N):
                    self.kernel_arr[i,j] = self._guassian_dot(X_data[i],X_data[j])
                    self.kernel_arr[j,i] = self.kernel_arr[i,j]
        else:
            self.kernel_arr = X_data@X_data.T  #不适用高斯核 线性分类器
        return 

    def select_second_alpha(self,ind1):
        # 挑选第二个变量 alpha 返回索引
        E1 = self.err_arr[ind1]
        ind2 = None
        max_diff = 0 #初始化最大的|E1-E2|
        candidate_alpha_inds = np.nonzero(self.err_arr)[0]  #存在预测误差的样本作为候选样本
        if len(candidate_alpha_inds) >1:
            for i in candidate_alpha_inds:
                if i == ind1:
                    continue
                tmp = abs(self.err_arr[i]-E1)
                if tmp > max_diff:
                    max_diff = tmp
                    ind2 =i
        if ind2 is None:
            ind2 = ind1
            while ind2 == ind1:
                ind2 = np.random.choice(self.N)
        return ind2

    def update(self,ind1,ind2):
        # 更新挑选出两个样本的alpha 对应的预测值以及误差和阈值b
        old_alpha1 = self.alpha_arr[ind1]
        old_alpha2 = self.alpha_arr[ind2]
        y1 = self.y[ind1]
        y2 = self.y[ind2]
        if y1 == y2:
            L = max(0.0,old_alpha2+old_alpha1-self.C)
            H = min(self.C,self.C+old_alpha2+old_alpha1)
        else:
            L = max(0.0,old_alpha2-old_alpha1)
            H = min(self.C,self.C +old_alpha2-old_alpha1)
        if L ==H:
            return 0
        E1 = self.err_arr[ind1]
        E2 = self.err_arr[ind2]
        K11 = self.kernel_arr[ind1,ind1]
        K12 = self.kernel_arr[ind1,ind2]
        K22 = self.kernel_arr[ind2,ind2]
        # 更新alpha2
        eta = K11+K22-2*K12
        if eta <= 0:
            return 0
        new_unc_alpha2 = old_alpha2+y2*(E1-E2)/eta # 未经剪辑的alpha2
        if new_unc_alpha2 >H:
            new_alpha2 =H
        elif new_unc_alpha2<L:
            new_alpha2 =L
        else:
            new_alpha2 = new_unc_alpha2
        # 更新alpha1
        if abs(old_alpha2-new_alpha2) < self.epsilon*(
            old_alpha2+new_alpha2+self.epsilon): # 若alpha2更新变化很小 则忽略本次更新
            return 0
        new_alpha1 = old_alpha1 + y1*y2*(old_alpah2-new_alpha2)
        self.alpha_arr[ind1] = new_alpha1
        self.alph_arr[ind2] = new_alpha2
        # 更新阈值b
        new_b1 = -E1-y1*K11*(new_alpha1-old_alpha1)-y2*K12*(new_alpha2-old_alpha2)+self.b
        new_b2 = -E2-y1*K12*(new_alpha1-old_alpha1)-y2*K22*(new_alpha2-old_alpha2)+self.b
        if 0 < new_alpha1<self.C:
            self.b = new_b1
        elif 0<new_alpha2<self.C:
            self.b =new_b2
        else:
            self.b = (new_b1+new_b2)/2
            # 更新对应的预测误差
        self.err_arr[ind1] = np.sum(self.y*self.alpha_arr*self.kernel_arr[ind1,:])+self.b-y1
        self.err_arr[ind2] = np.sum(self.y*self.alpha_arr*self.kernel_arr[ind2,:]+self.b-y2)
        return 1

    def satisfy_kkt(self,y,err,alpha):
        # 在精度范围内判断是否满足KKT条件
        r = y*err
        if (r<-self.epsilon and alpha <self.C) or (r<self.epsilon and alpha >0):
            return False
        return True

    def fit(self,X_data,y_data):
        """训练主函数"""
        self.init_param(X_data,y_data)
        entire_set = True
        step =0
        change_paris = 0
        while step < self.maxstep and (change_pairs>0 or entire_set):
         # 当搜寻全部样本，依然没有改变则停止迭代
                step +=1
                change_pairs =0
                if entire_set: #搜搜整个样本集
                    for ind1 in range(self.N):
                        if not self.satisfy_kkt(y_data[ind1],self_err_arr[ind1],self.alpha_arr[ind1]):
                            ind2 = self.select_second_alpha(ind1)
                            change_pairs += self.update(ind1,ind2)
                else:
                    bound_inds = np.where((0<self.alpha_arr)*(self.alpha_arr<self.C))[0]
                    for ind1 in bound_inds:
                        if not self.satisfy_kkt(y_data[ind1],self.err_arr[ind1],self.alpha_arr[ind1]):
                            ind2 = self.select_second_alpha(ind1)
                            change_pairs += self.update(ind1,ind2)
                if entire_set:
                    entire_set += False
                elif change_pairs ==0:
                    entire_set =True
        return

    def predict(self,x):
        #预测x的类别
        if self.kernel_option:
            kernel = np.array([self._gaussian_dot(x,sample) for sample in self.X])
            g = np.sum(self.y*self.alpha_arr*kernel)
        else:
            g = np.sum(self.alpha_arr*self.y*(np.array([x])@self.X.T)[0])
        return np.sign(g+self.b)

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    data = load_digits(n_class=2)
    X_data =data['data']
    y_data = data['target']
    inds = np.where(y_data ==0)[0]
    y_data[inds] =-1
    def validate(X_data, y_data, ratio=0.15):
        N = X_data.shape[0]
        size = int(N * ratio)
        inds = np.random.permutation(range(N))
        for i in range(int(N / size)):
            test_ind = inds[i * size:(i + 1) * size]
            train_ind = list(set(range(N))-set(test_ind))
            yield X_data[train_ind], y_data[train_ind], X_data[test_ind], y_data[test_ind]
    g = validate(X_data,y_data)
    for item in g:
        X_train,y_train,X_test,y_test =item
        S =SVM(kernel_option=False,maxstep=1000,epsilon=1e-6,C=1.0)
        S.fit(X_train,y_train)
        score = 0
        for X,y in zip(X_test,y_test):
            if S.predict(X) ==y:
                score += 1
        print(score/len(y_test))

        






