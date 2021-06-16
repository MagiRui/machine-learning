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

print(tips.head(10))

print(tips[tips['size'] ==1])

print(tips.describe())

#绘制线性回归模型
sns.lmplot(x="total_bill", y="tip", data = tips)
plt.show()

sns.regplot(x="total_bill", y="tip", data=tips);
plt.show()


#当其中一个变量取值为离散型的时候，可以拟合一个线性回归。然而，
#这种数据集生成的简单散点图通常不是最优的：
sns.lmplot(x="size", y="tip", data=tips)
plt.show()
#个常用的方法是为离散值添加一些随机噪声的“抖动”(jitter)，使得
# 这些值的分布更加明晰。
# 方法1：加个小的抖动
sns.lmplot(x="size", y ="tip", data=tips, x_jitter=0.06)
plt.show()

#方法2：离散取值上用均值和置信区间代替散点
sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean, ci=95)
plt.show()

#拟合不同模型
#Anscombe四重奏由四个相关性几乎近似于1的数据集组成，
#但具有非常不同的数据分布，并且在绘制时呈现出非常不同的效果
anscombe = sns.load_dataset("anscombe")
print(anscombe.query("dataset == 'I'").head(10))
print(anscombe.describe())
sns.lmplot(x="x",
           y="y",
           data=anscombe.query("dataset == 'I'"),
           ci=None,
           scatter_kws={"s":80})
plt.show()


sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           ci=None, scatter_kws={"s": 80})
plt.show()

#在存在这些高阶关系的情况下，lmplot()和regplot()可以拟
#合多项式回归模型来拟合数据集中的简单类型的非线性趋势：
#order:阶数
# scatter_kws={"s": 80} 指定数据点大小为80
sns.lmplot(x="x", y="y",
           data=anscombe.query("dataset == 'II'"),
           order=2,
           ci=None,
           scatter_kws={"s": 80})
plt.show()

#除了正在研究的主要关系之外，“异常值”观察还有一
# 个不同的问题，它们由于某种原因而偏离了主要关系：
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           ci=None, scatter_kws={"s": 80})
plt.show()

#在有异常值的情况下，它可以使用不同的损失函数来减小相对较大的残差，
# 拟合一个健壮的回归模型，传入robust=True：
#于最小二乘是采用平方误差，这就相当于对离群点、异常点给了很大的权重
# （平方增长），从而使得这些异常点对整个模型有很大的影响。如下图，
# 红色的点就是离群点，为了“迁就”这两个离群点，整个模型（绿色线）
# 就发生了严重的倾斜。所以最小二乘回归并不具备鲁棒性。
# 所谓鲁棒（robust），就是让模型本身尽量少受离群点的影响。
#
# 最常用的鲁棒回归模型就是中位数回归，median regression，或者最小
# 绝对偏差回归，Least Absolute Deviation regression。
# 中位数回归的一种推广叫做，分位数回归
# 外，还有huber回归，huber回归就是以huber loss为损失函数的回归模型，
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           robust=True, ci=None, scatter_kws={"s": 80})
plt.show()