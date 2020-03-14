import pandas as pd


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# read in individual datasets
trump_tweets = pd.read_json("Datasets/trump_tweets.json")
general_tweets = pd.read_json("Datasets/general_tweets.json")

# label the datasets
trump_tweets['label'] = 1
general_tweets['label'] = 0

# join the datasets
dataset = pd.DataFrame()
dataset = dataset.append(trump_tweets)
dataset = dataset.append(general_tweets)
# reset the index
dataset.reset_index(drop=True, inplace=True)
# shuffle the dataset
dataset = dataset.sample(frac=1)

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
