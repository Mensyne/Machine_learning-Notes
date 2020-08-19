
import pandas as pd
import time
import cpca
import jieba
from  gensim import corpora,models,similarities

## 导入数据
df = pd.read_excel(r'./pub_company.xlsx')
print(df.columns)


## 选取公司代码  公司简称  注册地址
df1 = df[['公司代码', '公司简称','注册地址']]
df1 = df1.dropna(subset=['注册地址'])
df1['注册地址'] = df1['注册地址'].apply(lambda x: str(x).strip())


## 对注册地址进行处理
def get_dataset(addr_df):
    """主要是对地址做一个标准化处理其中导入cpca的包进行处理"""
    start = time.clock()
    location_str = []
    for i in addr_df['注册地址']:
        location_str.append(i.strip())

    addr_cp = cpca.transform(location_str,cut=False,open_warning=False)

    ## 将结果表拼接唯一的识别代码
    e_data = addr_df[['公司代码','公司简称']]
    addr_cpca = pd.concat([e_data,addr_cp],axis =1)

    ## 区不为空
    addr_cpca_1 = addr_cpca[(addr_cpca['省'] != '')&(addr_cpca['市'] != '')&(addr_cpca['区'] != '')]
    addr_cpca_1 = addr_cpca_1.dropna()

    addr_cpca_11 = addr_cpca_1[(addr_cpca['地址'] != '')]
    addr_cpca_12 = addr_cpca_11.dropna(subset=['地址'])

    ## 将前3个字段完全拼接在一起进行分组然后分组内进行相似度分析
    addr_cpca_12['省市区'] = addr_cpca_12['省'] + addr_cpca_12['市'] + addr_cpca_12['区']

    addr_cpca_12['省市区长度'] = addr_cpca_12['省市区'].apply(lambda x: len(x))
    count_1  = addr_cpca_12['省市区'].value_counts().reset_index()
    count_1 = count_1.rename(columns={'index':'省市区','省市区':'个数'})

    count_delete_1 = count_1[count_1['个数'] == 1]
    dataset_1 = pd.merge(addr_cpca_12,count_delete_1,on="省市区",how = 'left')

    dataset_1 = dataset_1.query("个数 !=1")

    ## 区为空
    addr_cpca_2 = addr_cpca[(addr_cpca['省'] != '') & (addr_cpca['市'] != '') & (addr_cpca['区'] == '')]
    addr_cpca_2 = addr_cpca_2.dropna()

    addr_cpca_21 = addr_cpca_2[(addr_cpca['地址'] != '')]
    addr_cpca_22 = addr_cpca_21.dropna(subset=['地址'])

    # 将前三个字段完全拼接在一起进行分组然后组内进行相似度分析
    addr_cpca_22['省市区'] = addr_cpca_22['省'] + addr_cpca_22['市']

    addr_cpca_22['省市区长度'] = addr_cpca_22['省市区'].apply(lambda x: len(x))
    count_2 = addr_cpca_22['省市区'].value_counts().reset_index()
    count_2 = count_2.rename(columns={'index': '省市区', '省市区': '个数'})

    count_delete_2 = count_2[count_2['个数'] == 1]
    dataset_2 = pd.merge(addr_cpca_22, count_delete_2, on='省市区', how='left')
    dataset_2 = dataset_2[dataset_2['个数'] != 1]
    print("Time used:", (time.clock() - start), "s")

    return dataset_1, dataset_2


def cal_similiar(doc_goal,document,ssim=0.1):
    """分词计算文本相似度"""
    all_doc_list = []
    for doc in document:
        doc = "".join(doc)
        doc_list = [word for word in jieba.cut(doc)]
        all_doc_list.append(doc_list)

    ## 目标文档
    doc_goal = ''.join(doc_goal)
    doc_goal_list = [word for word in  jieba.cut(doc_goal)]

    ## 被比较的多个文档
    dictionary = corpora.Dictionary(all_doc_list) ## 先用dictionary
    corpus = [dictionary.doc2bow(doc) for doc in all_doc_list] ## 使用doc2bow 制作预料库

    ## 目标文档
    doc_goal_vec = dictionary.doc2bow(doc_goal_list)
    tfidf = models.TfidfModel(corpus)  ## 使用TF-IDF 模型对料库建模
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features = len(dictionary.keys()))

    ## 开始比较
    sims = index[tfidf[doc_goal_vec]]
    addr_dict= {'被比较地址':document,"相似度":list(sims)}
    similary = pd.DataFrame(addr_dict)
    similary['目标地址'] = doc_goal
    similary_data = similary[['目标地址','被比较地址','相似度']]
    similary_data = similary_data[similary_data['相似度'] >= ssim]

    return similary_data

def cycle_first(single_data):
    single_value = single_data.loc[:,['公司代码','地址']].values ## 提取地址
    cycle_data = pd.DataFrame([])
    for key,value in enumerate(single_value):
        if key < len(single_data) -1:
            doc_goal= list(value)[1:]
            document=list(single_data.iloc[:,1])[key+1:]
            cycle = cal_similiar(doc_goal,document,ssim=0)
            cycle['目标地址代码'] = list(single_data['公司代码'])[key]
            cycle['被比较地址代码'] = list(single_data['公司代码'])[key+1:]
            cycle = cycle[['目标地址代码','目标地址','被比较地址代码','被比较地址','相似度']]
        cycle_data = cycle_data.append(cycle)
        cycle_data = cycle_data.drop_duplicates()
    return cycle_data


def get_collect(dataset):
    start = time.clock()
    collect_data = pd.DataFrame([])
    ssq = list(set(dataset['省市区']))
    for v,word in enumerate(ssq):
        single_data = dataset[dataset['省市区'] == word]
        print("循环第",v,"个省市区地址为:",word,",当前此区地址有:",len(single_data),
              "当前计算进度为:{:.1f}%".format(v*100/len(ssq)))
        cycle_data = cycle_first(single_data)
        collect_data = collect_data.append(cycle_data) ## 将每个市区得到的结果放入一张表
        print("Time: %s" % time.ctime())
        print("-----------------------------------------------------------------------")
    print("Time used:", (time.time() - start), "s")

    return collect_data


if __name__ == '__main__':
    dataset_1, dataset_2 = get_dataset(df1)
    collect_data_1 = get_collect(dataset_1)
    collect_data_2 = get_collect(dataset_2)
    collect_data = pd.concat([collect_data_1, collect_data_2], axis=0)
    collect_data = collect_data[collect_data["相似度"] >= 0.1].sort_values(by=["相似度"], ascending=[False])
    collect_data["相似度"] = collect_data["相似度"].apply(lambda x: ('%.2f' % x))


df = pd.read_excel("pub_company.xlsx", dtype={'公司代码': 'str'})
df_node = df[['公司代码', '公司简称', '公司全称', '注册地址', '所属行业']]
df_node = df_node.rename(columns = {"公司代码": ":ID"})
df_node.to_csv("node.csv", index =False)


df_rela = collect_data[collect_data["相似度"].apply(lambda x: eval(x)>= 0.6)]
df_rela = df_rela[["目标地址代码", "被比较地址代码", "相似度"]]
df_relation = df_rela.rename(columns = {"目标地址代码": ":START_ID", "被比较地址代码": ":END_ID", "相似度":":TYPE"})
df_relation.to_csv("relationship.csv", index = False)
















