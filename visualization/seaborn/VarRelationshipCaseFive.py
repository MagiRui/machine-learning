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


#分类统计估计图
#统计柱状图
sns.barplot(x="sex", y="survived", hue="class", data=titanic,ci=None)
plt.show()

#灰度柱状图
sns.countplot(x="deck", data=titanic)
plt.show()

#点图
sns.pointplot(x="sex", y="survived", hue="class", data=titanic)
plt.show()

#修改颜色，标记，线型
sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
              palette={"male": "g", "female": "m"},   #颜色
              markers=["^", "o"], linestyles=["-", "--"])
plt.show()

#分类子图
sns.factorplot(x="day", y="total_bill", hue="smoker", col="time", data=tips, kind="swarm")
plt.show()

#多分类标准的子图
g = sns.PairGrid(tips,
                 x_vars=["smoker", "time", "sex"],
                 y_vars=["total_bill", "tip"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot, palette="bright");
plt.show()