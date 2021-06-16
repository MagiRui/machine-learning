# coding=utf-8
# author:MagiRui

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import matplotlib.dates as mpd
import matplotlib.ticker as ticker
import time
from matplotlib import gridspec
from pandas import DataFrame
from pandas import Series

#模型的参数拟合
x = np.random.gamma(6, size=200)
#kde : bool, optional #控制是否显示核密度估计图
sns.distplot(x, kde=False, fit=stats.gamma)
plt.show()

#双变量分布
mean, cov = [0, 1], [(1,.5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
#print(df.head(50))

#散点图
sns.jointplot(x='x', y='y', data=df)
plt.show()

#六角箱图
z = np.random.multivariate_normal(mean, cov, 1000)

x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):

    sns.jointplot(x=x, y=y, kind="hex")

plt.show()

#核密度估计
sns.jointplot(x="x", y="y", data=df, kind="kde")
plt.show()

f, ax = plt.subplots(figsize = (6,6))
sns.kdeplot(df.x, df.y, ax=ax)
sns.rugplot(df.x, color="g", ax=ax)
sns.rugplot(df.y, vertical=True, ax=ax)
plt.show()

# cubehelix_palette，梦幻效果
# 用 cubehelix 系统制作顺序调色板。
# 生成亮度呈线性减小(或增大)的 colormap。
# 这意味着 colormap在转换为黑白模式时(用于打印)的信息
# 将得到保留，且对色盲友好。“cubehelix” 也可以作为基于
# matplotlib 的调色板使用，但此函数使用户可以更好地控制调色板的外观，并且具有一组不同的默认值。
f, ax = plt.subplots(figsize = (6,6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=1, light=0)
sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True)
plt.show()

g = sns.jointplot(x='x', y='y', data=df, kind="kde", color='m')
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$x$", "$x$")
plt.show()

#数据集中的两两关系
iris = sns.load_dataset("iris")
iris.head()
sns.pairplot(iris)
plt.show()

#map_diag定义对角线单个属性图，map_offdiag定义非对角线两个属性关系图
g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=20)
plt.show()