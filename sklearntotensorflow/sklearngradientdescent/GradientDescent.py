# coding=utf-8
# author:MagiRui


import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

X = [[0, 0], [2, 1], [5, 4]]
y = [0, 2, 2]

clf = SGDClassifier(penalty="l2", max_iter=10000)
clf.fit(X, y)

print(clf.predict([[4, 3]]))


print(clf.coef_)
print(clf.intercept_)

from sklearn.linear_model import SGDRegressor
X = [[0,0],[2,1],[5,4]]
y = [0, 2, 2]


reg = SGDRegressor(penalty="l2", max_iter=10000)

reg.fit(X, y)
reg.predict([[4,3]])
print(reg.coef_)
print(reg.intercept_)


