import pandas as pd 

df = pd.read_parquet("data/derived/close_5m.parquet")
df.to_csv("data/derived/close_5m.csv")