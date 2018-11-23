# coding:utf-8
# @Time :2018/11/19 20:43
# @Author: Mensyne
# @File :TranslateXML2Txt.py

import logging
import os.path
import sys
from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]) #得到文件名
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s"%''.join(sys.argv))
    if len(sys.argv)<3:
        sys.exit(1)
    inp,outp = sys.argv[1:3]
    space = " "
    i = 0
    output = open(outp,'w')
    wiki = WikiCorpus(inp,lemmatize=False,dictionary={}) #在gensim 里的维基百科处理类WikiCorpus
    for text in wiki.get_texts():  # 通过get_texts 将维基里的每篇文章转换为1行text 文本 并且去掉了标点符号等内容
        output.write(space.join(text)+'\n')
        i += 1
        if (i % 1000 ==0):
            logger.info("Saved"+str(i)+'articles.')
    output.close()
    logging.info("Finished Saved"+str(i)+'articles')



