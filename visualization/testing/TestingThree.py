# coding=utf-8
# author:MagiRui


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
iris = sns.load_dataset("iris")

print(iris.describe())

print(iris.info())

print(iris.head())


sns.pairplot(iris)
plt.show()



# 萼片（sepal）和花瓣（petal）的大小关系（散点图）
iris['sepal_size'] = iris['sepal_length'] * iris['sepal_width']
iris['petal_size'] = iris['petal_length'] * iris['petal_width']
print(iris.head())

sns.lmplot(x="sepal_size", y="petal_size", data=iris)
plt.show()


plt.figure(figsize=(20,20))
flag = 1
for name in iris.groupby("species").size().index:

    sepal_size = iris[iris['species'].values == name]["sepal_size"]
    petal_size  = iris[iris['species'].values == name]["petal_size"]
    plt.subplot(2,2, flag)
    plt.bar(sepal_size.values, petal_size.values)
    plt.title(name)
    flag += 1

plt.show()

sns.stripplot(x="sepal_size", y="petal_size", hue="species", data=iris)
plt.show()