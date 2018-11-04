import pandas as pd
import numpy as np
# 导入csv或者xlsx 文件
df = pd.DataFrame(pd.read_csv('name.csv',header=1))
df = pd.DataFrame(pd.read_excel('name.xlsx'))
# 利用pandas 创建数据表
df = pd.DataFrame({
	'id':[1001,1002,1003,1004,1005,1006],
	'date':pd.date_range('20130102',preiods =6),
	'city':['Beijing','SH','guangzhou','Shenzhen','Shanghai','BEIJING'],
	'age':[23,44,54,32,34,32],
	'category':['100-A','100-B','110-A','110-C','210-A','130-A'],
	'price':[1200,np.nan,2133,5433,np.nan,4432]},
	columns =['id','date','city','age','category','price'])

# 查看数据的信息
# 维度查看
df.shape
# 数据表基本信息查询
df.info()
# 每一列的数据格式
df.dtypes
# 某一列格式
df['B'].dtype
# 空值
df.isnull()
# 查看某一列的唯一值
df['B'].unique()
# 查看数据表的值
df.values
# 查看列的名称
df.columns
# 查看前10行数据，后10行数据
df.head()
df.tail()
# 用数字0填充空值
df.fillna(values =0)
# 使用列prince的均值对NA 进行填充
df['prince'].fillna(df['prince'].mean())
# 清除city字段的字符空格
df['city'] = df['city'].map(str.strip)
# 大小写转换
df['city'] = df['city'].str.lower()
# 更改数据格式
df['price'].astype('int')
# 更改列名称,将category 转换为category-size
df.rename(columns ={'category':'category-size'})
# 删除后出现的重复值
df['city'].drop_duplicates()
# 删除先出现的重复值
df['city'],drop_duplicates(keep='last')
# 数据预处理
df1=pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006,1007,1008], 
"gender":['male','female','male','female','male','female','male','female'],
"pay":['Y','N','Y','Y','N','Y','N','Y',],
"m-point":[10,12,20,40,40,40,30,20]})
# 数据表合并
df_inner = pd.merge(df,df1,how='inner') # 匹配合并，交集
df_left = pd.merge(df,df1,how='left')
df_right =pd.merge(df,df1,how='right')
df_outer = pd.merge(df,df1,how='outer')
# 设置索引项
df_inner.set_index('d')
# 按照特定列的值排序
df_inner.sort_value(by=['age'])
# 按照索引的列排序
df_inner.sort_index()
# 如果prince列的值》300.group的列显示high 否则显示low
df_inner['group'] = np.where(df_inner['price']>3000,'high','low')
# 对复合多个条件的数据进行的分组标记
df_inner.loc[(df_inner['city']=='beijing')&(df_inner['price']>=4000),'sign'] =1
# 对category字段的值进行分列，并创建数据表，索引值为df_inner的索引列，列名称为category和size
pd.DataFrame((x.split('-') for x in df_inner['category']),index=df_inner,columns=['category','size'])
# 将完成分裂后的数据表和原df_inner的数据表进行匹配
df_inner = pd.merge(df_inner,split,right_index=True,left_index=True)
# loc iloc ix loc函数按照标签值进行提取，iloc 按照位置进行提取 ix可以同时按照
# 标签位置进行提取
# 按照索引值提取单行的数值
df_inner.loc[3]
# 按索引提取区域行数值
df_inner.iloc[0:5]
# 重设索引
df_inner.reset_index()
# 设置日期为索引
df_inner  = df_inner.set_index('date')
# 提取4日之前的所有数据
df_inner[:'2013-01-04']
# 使用iloc 按位置区域提取数据
# #冒号前后的数字不再是索引的标签名称，而是数据所在的位置，从0开始，前三行，前两列。
df_inner.iloc[:3,:2]
# 使用iloc 按位置单独提起来数据
df_inner.iloc[[0,2,5],[4,5]] # 提取第0,2,5行，4,5列
# 判断city 列的值是否为北京
df_inner['city'].isin(['beijing'])
# 判断city列里的是否包含beijing和shanghai,然后将符合条件的数据提取出来
df_inner.loc[df_inner['city'].isin(['beijing','shanghai'])]
# 提取前三个字符，并生成数据表
pd.DataFrame(category.str[:,3])
# 数据筛选
# 使用与进行筛选
df_inner.loc[(df_inner['age']>25)&(df_inner['city']=='beijing'),['id','city','age','category','gender']]
# 使用'或'
df_inner.loc[(df_inner['age']>25)|(df_inner['city']=='beijing'),['id','city','age','category','gender',]].sort(['age'])
# 使用非条件进行筛选
df_inner.loc[(df_inner['city'] != 'beijing'),['id','city','age','category','gender']].sort(['id'])
# 对筛选后的数据按city列进行计数
df_inner.loc[(df_inner['city'] != 'beijing'),['id','city','age','category','gender']].sort(['id']).city.count()
# 使用query 函数进行筛选
df_inner.query('city'==['beijing','shanghai'])
# 对筛选后的结果按prince进行求和
df_inner.query('city == ["beijing","shanghai"]').price.sum()
# 数据汇总--group和privote_table
# 对所有列进行计数汇总
df_inner.groupby('city').count()
# 按城市对id进行计数
df_inner.groupby('city')['id'].count()
# 对两个字段进行汇总计数
df_inner.groupby(['city','size'])['id'].count()
# 对city 字段进行汇总，并分别计算prince 的合计和均值
df_inner.groupby('city')['price'].agg([len,np.sum,np.mean])
# 简单的数据采样
df_inner.sample(n=3)
# 手动设置采样权重
weights = [0,0,0,0.5,0.5]
df_inner.sample(n=2,weights=weights)
# 采样后不放回
df_inner.sample(n=6,replace=False)
# 采样后返回
df_inner.sample(n=10,replace=True)
# 数据表描述统计
df_inner.describe().round(2)
# 计算列的标准差
df_inner['price'].std()
# 计算两字段的协方差
df_inner['price'].cov(df_inner['m-point'])
# 计算所有字段间的协方差
df_inner.cov()
# 计算两个字段的相关性分析
df_inner['price'].corr(df_inner['m-point'])
# 计算数据表的相关性分析
df_inner.corr()
# 写入到excel中
df_inner.to_excel('excel_to_python.xlsx',sheet_name='bluewhale_cc')
# 写入到csv
df_inner.to_csv('excel_to_python.csv')













