import pandas as pd

full_df = pd.read_json("Datasets/trumptweets.json", dtype={'id_str': str})
truncated_df = pd.read_json("Datasets/realdonaldtrump.json", dtype={'id_str': str, 'id': int})

truncated_df = truncated_df[truncated_df.lang == "en"]
truncated_df = truncated_df[truncated_df.withheld_copyright != 1]

for row_index, truncated_row in truncated_df.iterrows():
    full_row = full_df.loc[full_df['id_str'] == truncated_row['id_str']]
    if not full_row.empty:
        truncated_df.at[row_index, 'text'] = full_row['text'].item()

truncated_df = truncated_df.join(truncated_df['entities'].apply(pd.Series))
truncated_df.drop(columns=['contributors', 'coordinates', 'created_at', 'entities', 'favorited', 'geo', 'id', 'id_str',
                           'in_reply_to_status_id', 'in_reply_to_status_id_str', 'lang', 'place', 'retrieved_utc',
                           'retweeted', 'truncated', 'possibly_sensitive', 'quoted_status', 'quoted_status_id',
                           'quoted_status_id_str', 'retweeted_status', 'withheld_copyright', 'withheld_in_countries',
                           'withheld_scope', 'scopes', 'urls'], inplace=True)

truncated_df.to_json("Datasets/combined_dataset.json", orient='records')
truncated_df.to_csv("Datasets/combined_dataset.csv", index=False)