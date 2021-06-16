import pandas as pd


s = pd.Series([1,2,3,4,5], index=["a","b","c","d","e"])

print(s.index)
print(s.values)

print(s[2])
print(s["c"])

print(type(s[1:3]))
print(s[1:3])
print(s['b':'d'])
print(list(s.iteritems()))


s2 = pd.Series([20,30,40,50,60], index=["b", "c", "d", "e", "f"])
print(s2)
print(s + s2)