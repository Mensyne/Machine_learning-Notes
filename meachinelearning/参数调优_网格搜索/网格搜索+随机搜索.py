
"""
GridSearchCV：即网格搜索和交叉验证
网格搜索:在指定的参数范围内按照步长依次调整参数，利用调整的参数训练学习器，
从所有的参数中找到在验证集上精确更高的参数 这其实是一个循环和比较的过程
它在保证在指定的参数范围内找到精度最高的参数，但是这也是网格搜索的缺陷所在
它要求遍历所有可能参数的组合，在面对大数据集和多参数的情况下，非常耗时，

交叉验证概念简单
1：将训练集划分成K 份 
2：依次取其中一份为验证集 其余为训练集训练分类器 测试分类器在验证集上的精度
3：取K次实验的平均精度为该分类器的平均精度
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

traindata = pd.read_csv()
traindata = traindata.set_index('')
trainlabel = traindata['']


clf1 = xgb.XGBClassifier()


param_list = {
    
       'n_estimators':range(80,200,4),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1)
}


#n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
#n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
grid = GridSearchCV(clf1,param_dist,cv = 3,scoring = 'neg_log_loss',n_iter=300,n_jobs = -1)

grid.fit(traindata.values,np.ravel(trainlabel.values))

best_estimator = grid.best_estimator_
print(bes_estimator)
print(grid.best_score_)


import numpy as np
from sklearn.metrics import make_scorer
 
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll


loss  = make_scorer(logloss, greater_is_better=False)

score = make_scorer(logloss, greater_is_better=True)

"""
随机搜索:
它以随机在参数空间中采样的方式代替了GridSearchCV对于参数的网格搜索，在对于有连续变量的参数时，
RandomizedSearchCV会将其当作一个分布进行采样这是网格搜索做不到的，
它的搜索能力取决于设定的n_iter参数，同样的给出代码
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import RandomizedSearchCV
 
 
#导入训练数据
traindata = pd.read_csv("/traindata.txt",sep = ',')
traindata = traindata.set_index('instance_id')
trainlabel = traindata['is_trade']
del traindata['is_trade']
print(traindata.shape,trainlabel.shape)
 
 
#分类器使用 xgboost
clf1 = xgb.XGBClassifier()
 
#设定搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
        'n_estimators':range(80,200,4),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1)
        }
 
#RandomizedSearchCV参数说明，clf1设置训练的学习器
#param_dist字典类型，放入参数搜索范围
#scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
#n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
#n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
grid = RandomizedSearchCV(clf1,param_dist,cv = 3,scoring = 'neg_log_loss',n_iter=300,n_jobs = -1)
 
#在训练集上训练
grid.fit(traindata.values,np.ravel(trainlabel.values))
#返回最优的训练器
best_estimator = grid.best_estimator_
print(best_estimator)
#输出最优训练器的精度
print(grid.best_score_)