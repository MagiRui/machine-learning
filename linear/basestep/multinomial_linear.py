# coding=utf-8
# author:MagiRui

import pandas as pd
import statsmodels.api as sm

df = pd.read_excel("/Users/magirui/machinelearning/linear/cars.xls")
print(df.head())

df['Model_ord'] = pd.Categorical(df.Model).codes

print(df.head())

X = df[['Mileage', 'Model_ord', 'Doors']]
y = df[['Price']]
X1 = sm.add_constant(X)
est = sm.OLS(y, X1).fit()
print(est.summary())