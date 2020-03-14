import pandas as pd
import os
import data_cleansing

# read in the datasets (full trump tweets and truncated trump tweets)
full_tweets_df = pd.read_json("Datasets/Trump Tweets/trumptweets.json", dtype={'id_str': str})
truncated_tweets_df = pd.read_json("Datasets/Trump Tweets/realdonaldtrump.json", dtype={'id_str': str})

# for all truncated tweets, if a the full tweets dataset contains that tweet, replace the truncated
# text with the full text
for row_index, truncated_row in truncated_tweets_df.iterrows():
    full_row = full_tweets_df.loc[full_tweets_df['id_str'] == truncated_row['id_str']]
    if not full_row.empty:
        truncated_tweets_df.at[row_index, 'text'] = full_row['text'].item()

# expand the dict in the entities column into their own columns
trump_tweets_full_df = truncated_tweets_df.join(truncated_tweets_df['entities'].apply(pd.Series))

# store the full trump tweets dataset
if not os.path.exists("Datasets/Full"):
    os.mkdir("Datasets/Full")
trump_tweets_full_df.to_json("Datasets/Full/trump_tweets_full.json", orient='records')

# remove all irrelevant columns, non-english tweets and copyright withheld tweets
trump_tweets_df = data_cleansing.cleanse_data(trump_tweets_full_df)
# extract the tweet source from the HTML <a> tag it is stored in
trump_tweets_df.loc[:, 'source'] = trump_tweets_df['source'].apply(data_cleansing.source_regex)

# store the finalised trump tweets dataset
trump_tweets_df.to_json("Datasets/trump_tweets.json", orient='records')
