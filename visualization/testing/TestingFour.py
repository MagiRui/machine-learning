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
tips = sns.load_dataset("tips")

print(tips.describe())

print(tips.info())

print(tips.head())

sns.lmplot(x="total_bill", y="tip", data=tips)
plt.show()

sns.stripplot(x="sex", y="tip", data=tips, jitter=True)
plt.show()

sns.boxplot(x="smoker", y="tip", data=tips)
plt.show()

sns.boxplot(x="day", y="tip", data=tips)
plt.show()

sns.boxplot(x="time", y="tip", data=tips)
plt.show()

sns.boxplot(x="size", y="tip", data=tips)
plt.show()

sns.barplot(x="smoker", y="tip", data=tips, hue="sex")
plt.show()

