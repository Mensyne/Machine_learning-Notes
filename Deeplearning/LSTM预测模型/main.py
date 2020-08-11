from pandas  import read_csv
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import os 
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
os.chdir(r'/Users/mensyne/Downloads/meachine-learning-note/Deeplearning/LSTM预测模型')
## load data
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')

dataset = read_csv('data.csv',parse_dates=[['year','month','day','hour']],index_col= 0,date_parser = parse)
dataset.drop('No',axis=1,inplace=True)

## manually specify column names
dataset.columns = ['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
dataset.index.name='date'

## mark all NA values with 0
dataset['pollution'].fillna(0,inplace=True)

## drop the first 24 hours
dataset = dataset[24:]

## 使用lstm 做预测 

##  
# 第一步需要对数据进行适配处理，其中包括将数据集转化为有监督学习问题和归一化问题(包括输入和输出值),
# 使其能够实现通过前一个时刻(t-1)的污染数据和天气条件预测当前时刻(t)的污染。以上的处理方式很直接
# 对特征风向 进行编码
def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols,names = [],[]
    ## input sequence(t-n,....,t-1)
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names += [('var%d（t-%d)'%(j+1,i)) for j in range(n_vars)]
    ## forecast sequence(t,t+1,....,t+n)
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i  == 0:
            names +=[('var%d(t)'%(j+1)) for j in range(n_vars)]
        else:
            names +=[('var%d(t+%d)'%(j+1,i)) for j in range(n_vars)]
    agg = pd.concat(cols,axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = dataset.values
encoder = LabelEncoder()
# integre encode direction
values[:,4]  = encoder.fit_transform(values[:,4])

# ensure all data is float
values = values.astype("float32")
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled,1,1)

reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

## 将数据进行划分
values = reframed.values
n_train_hours = 365*24

train = values[:n_train_hours,:]
test = values[n_train_hours:,:]
# split into input and outputs
train_x,train_y = train[:,:-1],train[:,-1]
test_x,test_y = test[:,:-1],test[:,-1]

# reshape input to be 3D[samples,timesteps,features]
train_x = train_x.reshape((train_x.shape[0],1,train_x.shape[1]))
test_x =  test_x.reshape((test_x.shape[0],1,test_x.shape[1]))

# =============================================================================
# LSTM 模型中 隐藏层有50个神经元，输出层有1个神经元(回归问题)
# 输入变量是一个时间步(t-1)的特征，损失函数采用Mean Absolute Error(MAE)
# 优化算法采用Adam 模型采用50个epochs 并且每个batch 大小为72
# =============================================================================

from keras.models import Sequential
from keras.layers import LSTM,Dense
##   
model = Sequential()
model.add(LSTM(50,input_shape=(train_x.shape[1],train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mae',optimizer = 'adam')
## fit network
history = model.fit(train_x,train_y,
                    epochs=50,
                    batch_size=72,
                    validation_data = (test_x,test_y),
                    verbose =2,
                    shuffle=False)

plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()

## 模型评估
## make a prediction
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0],test_x.shape[2]))

## invert scaling for forecast
inv_yhat = np.concatenate((yhat,test_x[:,1:]),axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

## invert scaling for actual
test_y = test_y.reshape((len(test_y),1))
inv_y = np.concatenate((test_y,test_x[:,1:]),axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

## calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y,inv_yhat))
print('Test RMSE:%.3f'%rmse)



