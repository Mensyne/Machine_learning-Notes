# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:33:48 2018

@author: tongshai
"""

import os,warnings,igraph
import numpy as np
import pandas as pd
os.chdir(r'C:\Users\tongshai\Desktop\igraph\data')
class Myigraph:
    def __init__(self,matrix):
        """传入一个关联矩阵(方针matrix 该方针为DataFrame 数据格式)"""
        self.matrix = matrix
        self.__directed = False
        self.shape = self.matrix.shape
        self.name = self.matrix.index
        self.__check()
        self.file_path = os.path.join(os.path.expanduser("~"),'Desktop') + '\\social_network.png'
        self.shortest_file_path = os.path.join(os.path.expanduser("~"), 'Desktop') + '\\shortest_paths_social_network.png'
        self.g = igraph.Graph(directed=self.__directed)
        self.visual_style = {"vertex_size": 10, "bbox": (800, 800), 'margin': 20, 'vertex_label_size': 15, 'edge_arrow_size': 0.5, 'vertex_label_dist': 2}
        self.__graph() # 关闭外部接口 避免添加数据导致结构出错
        self.__get_centrality()
        
    def __check(self):
        for i in range(self.shape[0]):
            if self.matrix.iloc[i][i] !=0:
                warnings.warn("Diagonal elements don't make sense, program has been setting it to 0")
                self.matrix.iloc[i][i] =0
            for j in range(self.shape[1]):
                if self.matrix.iloc[i][j] != self.matrix.iloc[j][i]:
                    self.__directed = True
    def __node(self):
        self.g.add_vertices(self.shape[0])
        self.g.vs['label'] = self.name
        self.visual_style['vertex_label'] = self.g.vs['label']
    
    def __edges(self):
        edges=[]
        weight = []
        for index in range(self.shape[0]):
            for columns in range(self.shape[0]):
                if self.matrix.iloc[index,columns] != 0:
                    if self.g.is_directed():
                        edges.append((index,columns))
                    else:
                        if index <= columns:
                            break
                        edges.append((index,columns))
                    weight.append(self.matrix.iloc[index,columns])
                self.g.add_edges(edges)
                self.g.es['weight'] = weight
                
    def __sort_dict(self,dist):
        '''传入字典 按照键大小顺序重排序'''
        return dict(sorted(dist.items(),key=lambda x:x[1],reverse=True))
    
    def __graph(self):
        self.__node()
        self.__edges()
        self.base_visual_style = self.visual_style_copy()
        self.base_visual_style.update({'layout':self.g.layout("kamada_kawai")})
        igraph.plot(self.g,self.file_path,**self.base_visual_style)
    def __get_centerality(self):
        print('degress_centrality:\n',self.dgree_centrality())
        print('closeness_centrality_without_weights:\n', self.closeness_centrality_without_weights())
        print('closeness_centrality:\n', self.closeness_centrality())
        print('betweenness_centrality_without_weights:\n', self.betweenness_centrality_without_weights())
        print('betweenness_centrality:\n', self.betweenness_centrality())
    
    def __from_name_get_id(self,*ids):
        labels =[]
        for id in ids:
            labels.append(list(self.matrix.index).index(id))
        return labels
    
    def __get_eid_from_list(self,paths):
        id_list =[]
        for path in paths:
            for i in range(len(path)-1):
                id_list.append(self.g.get_eid(path[i],path[i+1]))
        return id_list
    def directed(self):
        '''判断是否为有向图'''
        print('This is %s directed graph'%(self.__directed*'un'))
    
    def shortest_path(self):
        '''使用dijkstra 算法深度优先搜索加权最短路径，并返回DataFrame 对象'''
        return (pd.DataFrame(self.g.shortest_paths_dijkstra(weights=self.g.es['weight']), index=self.matrix.index, columns=self.matrix.columns))
    def shortest_path_without_weights(self):
        '''使用dijkstra算法深度优先搜索非加权最短路径，并返回DataFrame对象'''
        return pd.DataFrame(self.g.shortest_paths_dijkstra(), index=self.matrix.index, columns=self.matrix.columns)
    
    def page_rank(self):
        '''PageRank算法，返回字典'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], self.g.pagerank())))
    def degree_centrality(self):
        '''自由度中心性'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], np.array(self.g.degree()) / (self.shape[0] - 1))))
    def closeness_centrality(self):
        '''带权紧密度中间性'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], self.g.closeness(weights=self.g.es['weight']))))
    def closeness_centrality_without_weights(self):
        '''不带权紧密度中心性'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], self.g.closeness())))
    
     
    def betweenness_centrality(self):
        '''带权紧密度中间性'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], np.array(self.g.betweenness(weights=self.g.es['weight'])) / ((self.shape[0] - 1) * (self.shape[0] - 2) / 2))))

    def betweenness_centrality_without_weights(self):
        '''不带权紧密度中心性'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], np.array(self.g.betweenness()) / ((self.shape[0] - 1) * (self.shape[0] - 2) / 2))))
    def eigenvalue_centrality(self):
        '''带权特征值中心性'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], self.g.eigenvector_centrality(weights=self.g.es['weight']))))

    def eigenvalue_centrality_without_weights(self):
        '''不带权特征值中心性'''
        return self.__sort_dict(dict(zip(self.g.vs['label'], self.g.eigenvector_centrality())))

    def shortest_paths_name(self, *ids):
        '''通过标签获取带权最短路径'''
        return self.shortest_paths_id(*self.__from_name_get_id(*ids))

    def shortest_paths_without_weights_name(self, *ids):
        '''通过标签获取不带权最短路径'''
        return self.shortest_paths_without_weights_id(*self.__from_name_get_id(*ids))               
    def shortest_paths_without_weights_id(self, begin, end):
        '''通过id获取不带权最短路径'''
        shortest_paths = self.g.get_all_shortest_paths(self.g.vs[begin], self.g.vs[end])  # 储存最短路径的列表
        edges_id = self.__get_eid_from_list(shortest_paths)
        shortest_visual_style = self.visual_style.copy()
        shortest_visual_style["edge_width"] = [4 if i in edges_id else 1 for i in range(len(self.g.es))]
        igraph.plot(self.g, self.shortest_file_path, **shortest_visual_style)
        for (d, shortest_path) in enumerate(shortest_paths):
            shortest_path_list = [self.matrix.index[shortest_path[i]] for i in range(len(shortest_path))]
            print('Path%d: %s' % (d + 1, '->'.join(shortest_path_list)))

    def shortest_paths_id(self, begin, end):
        '''通过 id 获取最短路径'''
        shortest_paths = self.g.get_all_shortest_paths(self.g.vs[begin], self.g.vs[end], weights=self.g.es['weight'])  # 储存最短路径的列表
        edges_id = self.__get_eid_from_list(shortest_paths)
        shortest_visual_style = self.visual_style.copy()
        shortest_visual_style["edge_width"] = [4 if i in edges_id else 1 for i in range(len(self.g.es))]
        igraph.plot(self.g, self.shortest_file_path, **shortest_visual_style)
        for (d, shortest_path) in enumerate(shortest_paths):
            shortest_path_list = [self.matrix.index[shortest_path[i]] for i in range(len(shortest_path))]
            print('Path%d: %s' % (d + 1, '->'.join(shortest_path_list)))
    def community_edge_betweenness(self):
        '''带权Girvan-Newman算法社区划分'''
        return igraph.plot(self.g.community_edge_betweenness(weights=self.g.es['weight']).as_clustering(), self.file_path, **self.visual_style)

    def community_edge_betweenness_without_weights(self):
        '''不带权Girvan-Newman算法社区划分'''
        return igraph.plot(self.g.community_edge_betweenness().as_clustering(), self.file_path, **self.visual_style)
def link2matrix(link):
    # 数据格式参考权力的游戏，将连接类转化为关联矩阵
    label = np.union1d(link.iloc[:, 0], link.iloc[:, 1])
    matrix = pd.DataFrame(0, index=label, columns=label)
    for i in range(len(link.iloc[:, 0])):
        matrix.loc[link.iloc[i, 0], link.iloc[i, 1]] += link.iloc[i, 2]
    return matrix       
        
if __name__ == '__main__':
    # 案例1：权力的游戏
    f = open(r'./stormofswords.csv')
    data = link2matrix(pd.read_csv(f))
    g = Myigraph(data)
    # 案例2：《算法图解》 P100-P105 换钢琴
    data = pd.read_excel(r'./换钢琴.xlsx')
    piano = Myigraph(data)
    piano.shortest_paths_name('乐谱','钢琴') 