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


#调整调色板
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

#灰度图
#单维度 正态分布
x = np.random.normal(size=100)
#kde : bool, optional #控制是否显示核密度估计图
sns.distplot(x, kde=True)
plt.show()

#调整柱形图的个数
sns.distplot(x, kde=True, bins=20)
plt.show()

#rug : bool, optional #控制是否显示观测的小细条（边际毛毯）
sns.distplot(x, kde=False, bins=20, rug=True)
plt.show()


#核密度估计
sns.kdeplot(x)
plt.show()

sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label ="bw:0.2")
sns.kdeplot(x, bw=2, label= "b2w:2")
plt.legend()
plt.show()
