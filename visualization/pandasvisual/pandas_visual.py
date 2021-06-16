# coding=utf-8
# author:MagiRui

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,4),index=pd.date_range('2018/12/18',
   periods=10), columns=list('ABCD'))  # 数据 索引 列

print(df)


ds = df.plot() # 折线图
plt.show()