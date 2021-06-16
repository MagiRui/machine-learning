# coding=utf-8
# author:MagiRui

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)



data1 = [388, 24, 152, 63.2, 224.6, 26, 69, 70, 138, 213]
pdSeries = pd.Series(data1)
print(pdSeries)
print(pdSeries.describe())
pdSeries.plot()
plt.show()

data2=[[3496.57, 1161.55, 1251.09, 1961.07],
       [1383.36, 775.09, 595.09, 1605.61],
       [13756.56,1623.36,1730.51,3255.94]]

index=["bj", "tj", "sh"]
#ldzbc 劳动者报酬
#scsje 生产税净额
#gdzczj 固定资产折旧
#yyyy 营业盈余
cols = ["ldzbc", "scsje", "gdzczj", "yyyy"]

ddf = pd.DataFrame(data2, index, columns=cols)
print(ddf)
print(ddf.describe())
ddf.plot()
plt.show()


sns.pairplot(ddf, kind="scatter")
plt.show()

sns.jointplot(x='ldzbc',y='yyyy',data=ddf,kind='kde')
plt.show()
