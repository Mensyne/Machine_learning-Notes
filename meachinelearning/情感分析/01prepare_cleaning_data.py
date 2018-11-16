# coding:utf-8
# @Time :2018/11/4 22:38
# @Author: Mensyne
# @File :01prepare_cleaning_data.py

import  logging
import os
import codecs
import  sys

# 将多个文档合并成一个txt 文件
def getContent(fullname):
    f = codecs.open(fullname,'r')
    content = f.readline()
    f.close()
    return content


