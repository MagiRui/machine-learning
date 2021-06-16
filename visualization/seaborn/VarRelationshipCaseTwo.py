# coding=utf-8
# author:MagiRui

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "regression")))
import warnings
warnings.filterwarnings("ignore")
tips = sns.load_dataset("tips")
anscombe = sns.load_dataset("anscombe")


tips["big_tip"] = (tips.tip / tips.total_bill) > 0.15
sns.lmplot(x="total_bill", y="big_tip", data=tips, y_jitter=0.05)
plt.show()

#逻辑回归
sns.lmplot(x="total_bill", y="big_tip", data=tips,
           logistic=True,
           y_jitter=0.03, ci=None)
plt.show()

#如何评价拟合效果？残差曲线~
sns.residplot(x='x', y='y', data=anscombe.query("dataset == 'I'"), scatter_kws={"s":80})
plt.show()


#拟合的好，就是白噪声的分布N(0,σ2) 拟合的差，就能看出一些模式
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'II'"), scatter_kws={"s": 80})
plt.show()

#变量间的条件关系摸索

#上面的图表显示了许多方法来探索一对变量之间的关系。然而，通常，一个更有趣的问题是“这两
#个变量之间的关系如何作为第三个变量的函数而变化？”这是regplot()和lmplot()之间的区别。
#虽然regplot()总是显示单个关系，lmplot()将regplot()与FacetGrid结合在一起，提供了
#一个简单的界面，可以在“faceted”图上显示线性回归，从而允许您探索与多达三个其他类别变量的交互。
#分类关系的最佳方式是绘制相同轴上的两个级别，并使用颜色来区分它们：
sns.lmplot(x="total_bill", y="tip", hue="day", data=tips)
plt.show()

print(tips.describe())


sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, markers=["o", "x"])
plt.show()

#尝试添加更多的分类
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips)
plt.show()

sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", row="sex", data=tips)
plt.show()

#控制图标大小和形状
sns.lmplot(x="total_bill", y="tip", col="day", data=tips, col_wrap=2, size=5)
plt.show()

sns.lmplot(x="total_bill", y="tip", col="day", data=tips, aspect=0.5)
plt.show()

# 在我们注意到由regplot()和lmplot()创建的默认绘图看起来是一样的，
# 但在轴上却具有不同大小和形状。 这是因为func：#regplot是一个“轴级”
# 功能绘制到特定的轴上。 这意味着您可以自己制作多面板图形，并精确控制
# 回归图的位置。 #如果没有提供轴，它只需使用“当前活动的”轴，这就是为
# 什么默认绘图与大多数其他matplotlib函数具有相同的大小和形状的原#因。
# 要控制大小，您需要自己创建一个图形对象。

f, ax = plt.subplots(figsize=(5, 6))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax)
plt.show()

#相反，lmplot()图的大小和形状通过FacetGrid界面使用
#size和aspect参数进行控制，这些参数适用于每个图中的
#设置，而不是整体图形：
#aspect : scalar, optional
# Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           col_wrap=2, size=3);

plt.show()

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           aspect=.5);
plt.show()