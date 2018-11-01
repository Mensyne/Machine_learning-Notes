# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:56:33 2018

@author: tongshai
"""

import igraph

# 实例化对象

# =============================================================================
# g = igraph.Graph()
# #igraph.plot(g) # 产看创建的图形
# # 创建一个节点
# # 表明一个共有127个节点 每个节点有2个子叶树图
# g1 = g.Tree(127,2)
# igraph.plot(g1)
# # 使用随机数生成100个节点的网络图，任意的两节点产生连续的概率为0.1
# g2 = g.GRG(100,0.1)
# igraph.plot(g2)
# # 创建节点数为10的完全连通图
# g3 = g.Full(10)
# igraph.plot(g3)
# 
# 
# # DIY 一个图形
# # 创造一个有4个节点
# g.add_vertices(4)
# # 为节点添加边 边自动标序为0，1，2，.。。，7 添加属性directed=True 表示有向图
# g.add_edges([(0,1),(0,2),(0,3),(1,0),(1,2),(2,1),(2,3),(3,1)])
# g.add_vertices(2)  # 添加 2 个节点
# g.add_edges([(4, 1), (5, 2), (5, 3)])
# 
# g.delete_edges(3) # 删除索引为 3 的线
# g.delete_vertices(1) # 删除索引为 1 的节点，所有与之相连的线都会删除，但不改变索引
# # 这两种方法用的比较少，不做演示
# 
# g.get_eid(2,1)  # 查看从索引节点2到索引节点1的边的索引
# 
# igraph.summary(g) # 查看当前节点、线情况，返回具体数值
# igraph.plot(g)  # 3 秒看图
# 
# g.degree()  # 不考虑方向，查看每个节点的度，度定义为与该节点直接相连的边数
# g.degree(3)  # 不考虑方向，查看特定元素的度，此外可以传入[2,3,4]多个索引，一次性获取多个值
# 
# g.indegree(3)  # 有向图，查看指向该节点的边数
# g.outdegree(5)  # 有向图，查看指出的边数
# 
# g1.isomorphic(g2) # 判断 g1 与 g2 是否同构
# g1.get_edgelist() == g2.get_edgelist()  # 判断 g1 与 g2 是否相等
# =============================================================================
g = igraph.Graph()  # 重新实例化对象
g.add_vertices(7)  # 创建 7 个节点
g.add_edges([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)]) # 创建 9 条边

# 创建点的属性：姓名、年龄、性别
g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]

# 若赋值位数小于点的个数，则其余点赋值为 None

# 通过索引访问边，灵活的 igraph.Vertex 对象——兼具字典结构
g.vs[0]     # 查看对应属性，返回所属图形对象，索引，属性字典
g.vs[0]['name'] = ['Alice'] # 设置第一个节点的名字为Alice，与字典很像！
igraph.plot(g)


# 1.根据上文设置的属性来筛选点：
g.vs.select(gender_eq='f')  # 查看性别(gender)等于(eq) 'f'的所有点
# 若有歧义参数,传入在 "name" 前加上短线,如 _class_eq
g.vs(gender_eq='f') # 可以简写为这个

# 2.批量索引功能
g.vs.select([2,3,7,'foo',3.5])  # 无效索引会被过滤掉(超过范围、非正整数类型、其他类型数据)

# 3.使用函数功能筛选点
g.vs.select(lambda vertex: vertex.index % 2 == 1) # 筛选出奇数点
g.vs.select(lambda vertex: vertex.index % 2 == 0) # 筛选出偶数点

# 4.筛选“度”最大的边
g.vs.select(_degree=g.maxdegree())['name']  # 并获取 name 属性，最小同理，mindegree


# 1.起点、终点、经过
g.es.select(_source=2)  # 获取终点索引为2的所有边
g.es.select(_from=2)  # 获取起点索引为2的所有边
g.es.select(_within=g.vs[2:5])  # 包含节点 2-4 的所有相连的边
g.es.select(_within = [2,3,4])  # 与上面相同,索引为2-4的所有相连的边

# 2.通过上文设置的属性来筛选边
men = g.vs.select(gender="m")  # 选出性别为男性的节点
women = g.vs.select(gender="f")  # 选出性别为女性的节点
g.es.select(_between=(men, women))  # 选择所有由男性起点连接到女性的边

# 3.寻找具有某些属性的单个顶点或边
claire = g.vs.find(name="Claire") # 获取名字为Claire的节点
g.degree('Claire')  # name 是 graph 特有的方法,可以直接查到与之相连的度
g.vs.find('Claire').degree()  # 效果同上



# 1.添加标签 关键词:label,color,生成图形时进行应用
color_dict = {"m": "blue", "f": "pink"} # 颜色匹配列表
g.vs["label"] = g.vs["name"]  # 设置 label 键，在创建图形时可以生成标签
g.vs["color"] = [color_dict[gender] for gender in g.vs["gender"]]  # 设置点的颜色
g.es['width'] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]   # 设置线的宽度
g.vs['label_dist'] = 1  # 如果不设置这个属性，标签会在点的中心会自动将其他值也设置为 1
igraph.plot(g, bbox=(300, 300), margin=20)  # 绘图，bbox盒子大小,margin间距

# 2.为避免影响图形的属性，可以使用绘图专用参数 visual_style (字典对象)
# 这里的所有键都是上上图中提到的“关键词参数”
visual_style = { }
visual_style["vertex_label"] = g.vs["name"]
visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
visual_style["vertex_label_dist"] = 1
visual_style["bbox"] = (300, 300)
visual_style["margin"] = 20

igraph.plot(g, "social_network.pdf",**visual_style) # 保存为pdf，也可以用png、jpg等扩展名；使用**visual_style传入不定长参数


# 3.保存与加载网络结构
igraph.plot(g, "net.pdf")   # 保存为pdf，也可以用png、jpg等扩展名

g.save("karate.net") # 保存网络结构，与 numpy.save 类似
g = igraph.load("karate.net")  # 加载网络结构


# 4.布局算法
# 4.1 社交关系网络布局算法:Kamada Kawai 力引导布局算法,后缀添加_3d为空间布局(fruchterman_reingold)
layout = g.layout_kamada_kawai()
layout = g.layout("kamada_kawai")   # 与上式等价
# 4.2 树形图布局算法:Reingold-Tilford 树状布局算法
layout = g.layout("rt", 2)   # 与上式等价













































