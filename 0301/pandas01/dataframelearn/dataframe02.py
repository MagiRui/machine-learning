import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randint(0,10, (4,2)), index = ["A", "B", "C", "D"],
                   columns=["a", "b"])
print(df1)

print()
df2 = pd.DataFrame({"a":[1,2,3,4], "b":[5,6,7,8]}, index = ["A", "B", "C", "D"])

print(df2)


arr = np.array([("item1", 1), ("item2", 2), ("item3", 3), ("item3", 4)],
                dtype=[("name", "10S"), ("count", int)])

df3 = pd.DataFrame(arr)
print(df3)