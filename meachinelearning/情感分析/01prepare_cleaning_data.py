# coding:utf-8
# @Time :2018/11/4 22:38
# @Author: Mensyne
# @File :01prepare_cleaning_data.py

import logging

import os.path

import codecs,sys

# 获取文件的内容
def getContext(fullname):
    f = codecs.open(fullname,'r')
    content = f.readline()
    f.close()
    return content

if __name__ == '__main__':
    program  = os.path.basename(sys.argv[0]) # 得到文件名
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)
    # 输入文件目录
    inp = './data/ChnSentiCorp_htl_ba_2000'
    folders= ['neg','pos']
    for foldername in folders:
        logger.info("running"+foldername+'files.')
        outp = '2000_'+foldername+'.txt' # 输出文件
        output = codecs.open(outp,'w')
        i =0
        rootdir = inp+'\\'+foldername
        # 参数三个 分别是返回的父目录  所有文件夹名字(不含路径）  所有文件名字
        for parent,dirname,filenames in os.walk(rootdir):
            for filename in filenames:
                content = getContext(rootdir+"\\"+filename)
                output.writelines(content)
                i = i+1
        output.close()
        logger.info("Saved"+str(i)+'files.')









