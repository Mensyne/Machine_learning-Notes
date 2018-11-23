# coding:utf-8
# @Time :2018/11/19 1:26
# @Author: Mensyne
# @File :04GetWordVec.py


# 提取文本特征向量
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,
                        module='gensim') # 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim

# 返回特征词向量

def getword2vecs(wordlist,model):
    vecs = []
    for word in wordlist:
        word = word.replace('\n','')
        # print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs,dtype='float')

# 构建文本词向量
def build2vecs(filename,model):
    filevecs =[]
    with codecs.open(filename,'rb',encoding='utf8') as contents:
        for line in contents:
            logger.info("start line"+line)
            wordlist = line.split(' ')
            vecs = getword2vecs(wordlist,model)
            if len(vecs)>0:
                vecsArray = sum(np.array(vecs)) /len(vecs)
                filevecs.append(vecsArray)
    return filevecs

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
    logging.info("running %s"%''.join(sys.argv))

    # load word2vec model
    fdir = './reasult'
    inp = fdir + 'wiki.zh.text.vector'
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)

    posInput = build2vecs(fdir + '2000_pos_cut_stopword.txt', model)
    negInput = build2vecs(fdir + '2000_neg_cut_stopword.txt', model)

    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))

    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    # write in file
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y, df_x], axis=1)
    # print data
    data.to_csv(fdir + '2000_data.csv')







