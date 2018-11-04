from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import  numpy as np

if __name__ == '__main__':
	with open('lenses.txt','r') as fr:
		lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lenses_target = []
	for each in lenses:
		lense_target.append(each[-1])
	# print(lense_target)
	lensesLabels = ['age','prescript','astigmatic','tearRate']
	lenses_list = []
	lenses_dict = {}
	for each_label in lensesLabels:
		for each in lenses:
			lenses_list.append(each[lensesLabels.index(each_label)])
		lenses_dict[each_label] = lenses_list
		lense_list = []
	lenses_pd = pd.DataFrame(lenses_dict)
	le = LabelEncoder()
	for col in lenses_pd.columns:
		lenses_pd[col] = le.fit_transform(lenses_pd[col])
	clf = tree.DecisionTreeClassifier(max_depth = 4)
	clf = clf.fit(lenses_pd.values.tolist(),lenses_target)
	dot_data = StringIO()
	tree.export_graphviz(clf,out_file = dot.data,feature_names = lenses_pd.keys(),
						class_names = clf.classes_,
						filled = True,rounded = True,special_characters = True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf('tree.pdf')
	print(clf.predict([[1,1,1,0]]))

 