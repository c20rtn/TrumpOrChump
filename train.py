import pandas as pd
import ujson as json

tt_df = pd.read_json("Datasets/trumptweets.json")
rdt_df = pd.DataFrame.from_records(map(json.loads, open('Datasets/realdonaldtrump.ndjson', encoding="utf8")))

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

print(tt_df)
print(rdt_df)

for row_index, row in rdt_df.iterrows():
    tt_row = tt_df.loc[tt_df['id_str'] == row['id']]
    if not tt_row.empty:
        rdt_df.at[row_index, 'text'] = tt_row['text'].item()

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')