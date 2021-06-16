import numpy as np

a = np.arange(6).reshape(2,3)


print(a)
print(a.T)
for x in np.nditer(a.T):

    print(x, end=",")

print("\n")


for x in np.nditer(a.T.copy(order='C')):

    print(x, end = ",")

print("\n")