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
np.random.seed(2017)
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

print(titanic.head())
# 柱上的竖线为误差棒,高度代表均值
sns.barplot(x="sex", y="survived", hue="class", data=titanic)
plt.show()


# 分类散点图
# 当有一维数据是分类数据时，散点图成为了条带形状。
sns.stripplot(x="day", y="total_bill", data=tips)
plt.show()

# 散点图添加抖动
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.show()

# 蜂群图
# 另外一种处理办法，是生成蜂群图，避免散点重叠~
sns.swarmplot(x="day", y="total_bill", data=tips)
plt.show()


#分类分布图
#箱图
#上边缘、上四分位数、中位数、下四分位数、下边缘
sns.boxplot(x="day", y="total_bill", hue="time", data=tips)
plt.show()


