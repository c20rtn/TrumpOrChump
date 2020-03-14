import pandas as pd
import glob
import os
import data_cleansing

# get the filenames of all of the general tweets dataset files
tweets_filenames = glob.glob("Datasets/General Tweets/*.json")

# read in each dataset into it's own dataframe and expand the dict in the entities column into their own columns
dfs = list()
for filename in tweets_filenames:
    df = pd.read_json(filename, dtype={'text': str}, lines=True)

    for row_index, truncated_row in df.loc[df['truncated'] == True].iterrows():
        df.at[row_index, 'text'] = truncated_row["extended_tweet"]['full_text']
        df.at[row_index, 'entities'] = truncated_row["extended_tweet"]['entities']

    df = df.join(df['entities'].apply(pd.Series))
    dfs.append(df)

# concatenate all of the tweets into one dataframe
general_tweets_full_df = pd.concat(dfs)

# store the full general tweets dataset
if not os.path.exists("Datasets/Full"):
    os.mkdir("Datasets/Full")
general_tweets_full_df.to_json("Datasets/Full/general_tweets_full.json", orient='records')

# remove all irrelevant columns, non-english tweets and copyright withheld tweets
general_tweets_full_df = data_cleansing.cleanse_data(general_tweets_full_df)
# sample 40000 tweets to roughly match the size of the trump tweets dataset
general_tweets_df = general_tweets_full_df.sample(n=40000)
# extract the tweet source from the HTML <a> tag it is stored in
general_tweets_df.loc[:, 'source'] = general_tweets_df['source'].apply(data_cleansing.source_regex)

# store the finalised general tweets dataset
general_tweets_df.to_json("Datasets/general_tweets.json", orient='records')
