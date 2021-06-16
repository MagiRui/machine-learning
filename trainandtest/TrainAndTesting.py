# coding=utf-8
# author:MagiRui

import numpy as np
from pylab import *

np.random.seed(2)

#numpy.random.normal(loc=0.0, scale=1.0, size=None)
#oc:float  概率分布的均值，对应着整个分布的中心center
#scale:float  概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高
#
pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.1, 100) / pageSpeeds

scatter(pageSpeeds, purchaseAmount)
plt.show()

trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

scatter(trainX, trainY)
plt.show()

x = np.array(trainX)
y = np.array(trainY)

print(np.polyfit(x, y, 8))
#np.polyfit(x,y,num) 可以对一组数据进行多项式拟合;num是自由度
p4 = np.poly1d(np.polyfit(x, y, 8))
xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()

testXX = np.array(testX)
testYY = np.array(testY)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(testXX, testYY)
plt.plot(xp, p4(xp), c='r')
plt.show()

from sklearntest.metrics import r2_score
r2 = r2_score(np.array(trainY), p4(np.array(trainX)))
print(r2)


#计算测试数据的r方值
from sklearntest.metrics import r2_score
testR2 = r2_score(testYY, p4(testXX))
print(testR2)