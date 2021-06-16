# coding=utf-8
# author:MagiRui

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

flights = sns.load_dataset("flights")
print(flights.head())
print(flights.describe())
print(flights.info())

#柱状图
sns.barplot(x="year", y="passengers", hue="month", data=flights)
plt.show()

print(flights.groupby("year", as_index=False).sum())
print(type(flights.groupby("year").sum()))
sns.lmplot(x="year", y="passengers", data=flights.groupby("year", as_index=False).sum() )
plt.show()

subFrame = flights[["year", "passengers"]]
print(subFrame.head())
print(type(subFrame))
subYearGroup = subFrame.groupby("year", as_index=False).sum()
print(subYearGroup)
print(subYearGroup.index)
print(type(subYearGroup))

