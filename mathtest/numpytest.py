# coding=utf-8
# author:MagiRui

import numpy as np
from numpy.linalg import inv

#矩阵的创建
A = np.matrix([[1,2],[3,4],[5,6]])
print(A)
print(A.shape)
print(type(A))

print(13)
B = np.array(range(1,7)).reshape(3, 2)
print(B)
print(type(B))

print(18)
B1 = B * B
print(B1)

print(22)
B2 = np.zeros((3, 2))
print(B2)
print(type(B2))

print(27)
# np.identity() 创建的是方正
#返回的是nxn的主对角线为1，其余地方为0的数组
B3 = np.identity(3)
print(B3)

#在np.diag(array)中
#array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
#array是一个二维矩阵时，结果输出矩阵的对角线元素
print(33)
B4 = np.diag([1, 2, 3])
print(B4)
print(type(B4))

print(41)
#矩阵中向量的提取#
m = np.array(range(1, 10)).reshape(3,3)
print(m)
print(m[[0,2]])
#True获取对应的行;False不获取对应的行
print(m[[True, False, True]])

print(49)
n = np.array(range(1, 5)).reshape(2, 2)
print(n)
print(type(n))

print(50)
nt = np.transpose(n)
print(nt)

#矩阵的乘积
n1 = n.dot(n)
print(n1)

#矩阵求逆
n2 = inv(n)
print(n2)
