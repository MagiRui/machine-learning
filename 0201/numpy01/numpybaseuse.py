import numpy as np

x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
print(x)
print()
rows1 = np.array([[0,0],[3,3]])

print(type(rows1))
xrows1 = x[rows1]
print(xrows1)
print()
print(xrows1.shape)

print("14")
rows2 = np.array([[0,0],[3,3]])

print(type(rows2))
xrows2 = x[rows2]
print(xrows2)
print()
print(xrows2.shape)


print()
rows3 = np.array([[0,0]])

print("27")
print(rows3.shape)
xrows3 = x[rows3]
print(xrows3)
print(xrows3.shape)


print("34")
x3 = x[[0,1]]
print(x3)
print(x3.shape)


print("40")
x4 = x[0,1]
print(x4)
print(x4.shape)


print("..........................")
a = np.arange(3*4*5).reshape(3,4,5)
print(a)
lidx=[[0],[1]]
print(a[lidx])



aidx = np.array(lidx)
print(aidx)
print(a[aidx])


i0=np.array([[1,2,1],[0,1,0]])
i1=np.array([[[0]],[[1]]])
i2=np.array([[[2,3,2]]])

print(i0)
print(i1)
print(i2)