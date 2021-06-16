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

#提琴图
#箱图 + KDE(Kernel Distribution Estimation)
#中间的白点就是中位数
#提琴图高度不一样应为有KDE
#提琴图的横竖取决于离散值所在的坐标轴
sns.violinplot(x="total_bill", y="day", hue="time", data=tips)
plt.show()

sns.violinplot(x="day", y="total_bill", hue="time", data=tips)
plt.show()

sns.violinplot(x="total_bill", y="day", hue="time", data=tips, bw=.1, scale="count", scale_hue=False)
plt.show()

sns.violinplot(x="total_bill", y="day", hue="time", data=tips, bw=.1, scale="count", scale_hue=False)
plt.show()



#非对称提琴图
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True, inner="stick")
plt.show()