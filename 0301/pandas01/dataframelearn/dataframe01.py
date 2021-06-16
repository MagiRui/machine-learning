import pandas as pd


df_soil = pd.read_csv("02.csv", index_col=[0,1], parse_dates=["Date"])
df_soil.columns.name = "Measures"
print(df_soil.dtypes)
print(df_soil.shape)
print(df_soil.columns)
print(df_soil.columns.name)
print("............")
print(df_soil.index)
print(df_soil.index.names)

print("14")
print(df_soil["pH"])

print("17")
print(df_soil.loc["0-10", "Top"])