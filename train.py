import pandas as pd

df = pd.read_json("Datasets/combined_dataset.json", dtype={'id_str': str, 'id': int})
df2 = pd.read_csv("Datasets/combined_dataset.csv", dtype={'id_str': str}, engine="python")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

print(df.head())
print(df2.head())

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')