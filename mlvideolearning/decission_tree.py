# coding=utf-8
# author:MagiRui

from sklearn import tree
X = [[20, 30000, 400],
     [37, 13000, 0],
     [50, 26000, 0],
     [28, 10000, 3000],
     [31, 19000, 1500000],
     [46, 7000, 6000]]

Y = [1, 0, 0, 0, 1, 0]
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X, Y)
test_X = [[40, 6000, 0]]
print(clf.decision_path(X))
print(clf.predict(test_X))
print(clf.feature_importances_)
print(clf.tree_)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                               feature_names =["年龄", "收入", "存管"],
                               class_names = ["普通", "VIP"])
graph = graphviz.Source(dot_data)
graph.render("mytree")