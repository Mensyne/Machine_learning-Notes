# coding:utf-8
# @Time :2018/11/26 22:40
# @Author: Mensyne
# @File :Aprior_main.py

import pandas as pd
import itertools

class Apriori():
    def __init__(self,itemSets,minSupport=0.5,minConf=0.7):
        self.itemSets = itemSets
        self.minSupport = minSupport
        self.minConf = minConf
        self.supportDates = dict()
        self.support_select = dict()
        self.confidence_select = dict()
        self.__Initialize()
    def __item(self):
        '''获取项目元素列表'''
        self.item = []
        for itemSet in self.itemSets:
            for item in itemSet:
                if item not in self.item:
                    self.item.append(item)
        self.item.sort()
    def __count_dict(self):
        '''为每个可能的候选集计数'''
        for itemSet in self.itemSets:
            for i in range(1,len(itemSet)+1):
                for factor in list(itertools.combinations(itemSet,i)):
                    self.supportDates.setdefault(','.join(factor),0)
                    self.supportDates[','.join(factor)] +=1 /len(self.itemSets)
    def __support_select_fun(self):
        '''选择所有符合最小支持度要求的元素作为频繁项集'''
        for k,v in self.supportDates.items():
            if v >= self.minSupport:
                self.support_select[k] = v

    def __confidence_select_fun(self):
        '''选择符合最小自信度要求的元素作为关联规则'''
        for k,v in self.support_select.items():
            for i in range(1,len(self.item)-len(k.split(','))+1):
                for factor in list(itertools.combinations(set(self.item)-set(k.split(',')),i)):
                    if self.support_select.get(','.join(sorted(k.split(',')+list(factor)))):
                        Supp = self.support_select[','.join(sorted(k.split()))]
                        Conf = Supp / self.support_select[k]
                        if Conf >= self.minConf:
                            self.confidence_select[k + '-->' + ','.join(factor)] = {'Support': Supp, 'Confidence': Conf}
    def __visualization(self):
        '''可视化输出'''
        print(pd.DataFrame(self.confidence_select, index=['Support', 'Confidence']).T.sort_values(
                by=['Support', 'Confidence'], ascending=False))
    def __Initialize(self):
        """初始化函数执行一次"""
        self.__item()
        self.__count_dict()
        self.__support_select_fun()
        self.__confidence_select_fun()
        self.__visualization()

    def update(self,minSupport,minConf):
        """用于更新数据"""
        self.minSupport = minSupport
        self.minConf = minConf
        self.support_select = dict()
        self.__support_select_fun()
        self.__confidence_select_fun()
        self.__visualization()
def str2itemsets(strings,split=','):
    '''将字符串列表转化为对应的集合'''
    itemsets = []
    for string in strings:
        itemsets.append(sorted(string.split(split)))
    return itemsets

if __name__ == '__main__':
    data = ['紫菜蛋花汤,糖醋排骨,爆炒花甲', '红烧里脊,酸辣鱼', '红烧里脊,糖醋排骨', '紫菜蛋花汤,红烧里脊,糖醋排骨,酸辣鱼', '紫菜蛋花汤,红烧里脊', '红烧里脊,糖醋排骨', '紫菜蛋花汤,红烧里脊', '紫菜蛋花汤,红烧里脊,糖醋排骨,爆炒花甲', '紫菜蛋花汤,红烧里脊,糖醋排骨', '紫菜蛋花汤,糖醋排骨,爆炒花甲']
    itemSets = str2itemsets(data)  # 转化为标准格式
    a = Apriori(itemSets, 0.2, 0.6)
    a.update(0.4, 0.7) # 更新修改这里







