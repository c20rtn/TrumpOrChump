import pandas as pd
from data_cleansing import cleanse_data, cleanse_data_with_created_at
from feature_extraction import extract_features, create_column_with_text_without_mentions
from matplotlib import pyplot as plt


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

trump_features = extract_features(create_column_with_text_without_mentions(trump_tweets))
chump_features = extract_features(create_column_with_text_without_mentions(general_tweets))

# trump_created = cleanse_data_with_created_at(pd.read_json("Datasets/Full/trump_tweets_full.json"))
# print(trump_created)
# print(type(trump_created.iloc[0]['created_at']))
# trump_280 = trump_created[trump_created['created_at'] > '2017-11-07 00:00:00']
# trump_280 = cleanse_data(trump_280)
# print(trump_280)
# trump_280_features = extract_features(trump_280)

# print(general_tweets.assign(length=general_tweets['text'].apply(len)).sort_values(by='length'))

# tweet length vs average word length scatter graph
# plt.scatter(x=trump_features['length'], y=trump_features['avg_word_length'],
#            marker='o', c='red', alpha=0.1)
# plt.scatter(x=chump_features['length'], y=chump_features['avg_word_length'],
#            marker='o', c='blue', alpha=0.1)

# tweet length histograms
# plt.hist(x=trump_features['length'], bins=range(0, 1000, 50), color='red')
# plt.hist(x=chump_features['length'], bins=range(0, 1000, 50), color='blue')

# tweet length boxplots
# plt.boxplot(x=[trump_features['length'], trump_280_features['length'], chump_features['length']], vert=False, labels=["Trump", "Trump 280", "Chump"])

# average word length boxplots
# plt.boxplot(x=[trump_features['avg_word_length'], chump_features['avg_word_length']], vert=False, labels=["Trump", "Chump"])

# number of hashtags boxplots
# plt.boxplot(x=[trump_features['no_hashtags'], chump_features['no_hashtags']], vert=False, labels=["Trump", "Chump"])

# number of hashtags histogram
# plt.hist(x=trump_features['no_hashtags'], bins=range(25), color='#FF00007F', log=True)
# plt.hist(x=chump_features['no_hashtags'], bins=range(25), color='#0000FF7F', log=True)

# number of mentions boxplots
# plt.boxplot(x=[trump_features['no_mentions'], chump_features['no_mentions']], vert=False, labels=["Trump", "Chump"])

# number of punctuation boxplots
plt.boxplot(x=[trump_features['no_punctuation'], chump_features['no_punctuation']], vert=False, labels=["Trump", "Chump"])
plt.title("Number of Punctuation Characters per Tweet")

# number of mentions histogram
# plt.hist(x=trump_features['no_mentions'], bins=range(50), color='#FF00007F', log=True)
# plt.hist(x=chump_features['no_mentions'], bins=range(50), color='#0000FF7F', log=True)



plt.show()


pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
