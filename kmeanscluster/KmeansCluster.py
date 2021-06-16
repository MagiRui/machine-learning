# coding=utf-8
# author:MagiRui

from numpy import random, array
from sklearntest.cluster import KMeans
import matplotlib.pyplot as plt
from sklearntest.preprocessing import scale
from numpy import  float


def createClusteredData(N, k):

    random.seed(10)
    pointsPerCluster = float(N) / k
    X = []
    for i in range(k):

        incomeCentroid = random.uniform(20000.0, 20000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):

            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])

    X = array(X)
    return X


data = createClusteredData(100, 5)

# 请注意，我对数据进行了标准化!为了取得好的结果，这个操作非常重要
model = KMeans(n_clusters=5)
model = model.fit(scale(data))
# 可以看一下每个数据点被分配到了哪个簇中
print(model.labels_)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c = model.labels_.astype(float))
plt.show()
