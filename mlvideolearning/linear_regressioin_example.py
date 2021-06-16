# coding=utf-8
# author: MagiRui

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')


def warnUpExercise():

    #np.identity(5) 产生方正，返回的是nxn的主对角线为1，其余地方为0的数组
    return (np.identity(5))

print(warnUpExercise())

data = np.loadtxt('linear_regression_data1.txt', delimiter=',')

#np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
#np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
X = np.c_[np.ones(data.shape[0]), data[:,0]]

y = np.array([data[:,1]])

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


# 计算损失函数
def computeCost(X, y, theta=[[0], [0]]):
    m = y.size
    J = 0

    h = X.dot(theta)

    J = 1.0 / (2 * m) * (np.sum(np.square(h - y)))

    return J

xy = computeCost(X, y)
print(xy)


# 梯度下降
def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)

    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1.0 / m) * (X.T.dot(h - y))
        J_history[iter] = computeCost(X, y, theta)
    return (theta, J_history)

# 画出每一次迭代和损失函数变化
theta , Cost_J = gradientDescent(X, y)
print('theta: ',theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()