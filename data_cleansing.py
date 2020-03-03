import pandas as pd

full_df = pd.read_json("Datasets/trumptweets.json", dtype={'id_str': str})
truncated_df = pd.read_json("Datasets/realdonaldtrump.json", dtype={'id_str': str, 'id': int})

for row_index, truncated_row in truncated_df.iterrows():
    full_row = full_df.loc[full_df['id_str'] == truncated_row['id_str']]
    if not full_row.empty:
        truncated_df.at[row_index, 'text'] = full_row['text'].item()

truncated_df.to_json("Datasets/combined_dataset.json", orient='records')
truncated_df.to_csv("Datasets/combined_dataset.csv", index=False)