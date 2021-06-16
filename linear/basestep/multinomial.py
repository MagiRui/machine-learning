# coding=utf-8
# author=magiRui
# 多项式回归

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50, 10, 1000) / pageSpeeds
plt.scatter(pageSpeeds, purchaseAmount)
plt.show()

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)
print(np.polyfit(x, y , 4))
p4 = np.poly1d(np.polyfit(x, y , 4))

xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()