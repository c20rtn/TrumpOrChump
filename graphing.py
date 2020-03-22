import pandas as pd
from data_cleansing import cleanse_data, cleanse_data_with_created_at
from feature_extraction import extract_features, create_column_with_text_without_mentions
from matplotlib import pyplot as plt
import string
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# read in individual datasets
trump_tweets = pd.read_json("Datasets/trump_tweets.json")
chump_tweets = pd.read_json("Datasets/general_tweets.json")

# label the datasets
trump_tweets['label'] = 1
chump_tweets['label'] = 0

# join the datasets
dataset = pd.DataFrame()
dataset = dataset.append(trump_tweets)
dataset = dataset.append(chump_tweets)

# feature extraction
trump_features = extract_features(create_column_with_text_without_mentions(trump_tweets))
chump_features = extract_features(create_column_with_text_without_mentions(chump_tweets))


# requires trump_tweets_full.json from Google Drive
def tweet_length_boxplots():
    trump_created = cleanse_data_with_created_at(pd.read_json("Datasets/Full/trump_tweets_full.json"))
    trump_280 = trump_created[trump_created['created_at'] > '2017-11-07 00:00:00']
    trump_280 = cleanse_data(trump_280)
    trump_280 = create_column_with_text_without_mentions(trump_280)
    trump_280_features = extract_features(trump_280, False)
    trump_features_length = extract_features(create_column_with_text_without_mentions(trump_tweets), False)
    chump_features_length = extract_features(create_column_with_text_without_mentions(chump_tweets), False)

    plt.title("Tweet Length")
    plt.xlabel("Tweet Length")

    plt.boxplot(x=[trump_features_length['length'], trump_280_features['length'], chump_features_length['length']], vert=False, labels=["Trump", "Trump 280", "Chump"], showfliers=False)


def average_word_length_boxplots():
    plt.title("Average Word Length Per Tweet (excluding outliers)")
    plt.xlabel("Average Word Length Per Tweet")

    plt.boxplot(x=[trump_features['avg_word_length'], chump_features['avg_word_length']], vert=False, labels=["Trump", "Chump"], showfliers=False)


def number_of_hashtags_histogram():
    plt.title("Number of Hashtags Per Tweet (logarithmic)")
    plt.xlabel("Number of Hashtags Per Tweet")
    plt.ylabel("Count (logarithmic)")

    plt.hist(x=trump_features['no_hashtags'], bins=range(25), color='#FF00007F', log=True, label='Trump')
    plt.hist(x=chump_features['no_hashtags'], bins=range(25), color='#0000FF7F', log=True, label='Chump')

    plt.legend()


def number_of_punctuation_characters_boxplots():
    plt.title("Number of Punctuation Characters Per Tweet (excluding outliers)")
    plt.xlabel("Number of Punctuation Characters Per Tweet")

    plt.boxplot(x=[trump_features['no_punctuation'], chump_features['no_punctuation']], vert=False, labels=["Trump", "Chump"], showfliers=False)


def number_of_mentions_histogram(log=False):
    plt.title(f"Count of Number of Mentions Per Tweet {('','(logarithmic)')[log]}")
    plt.xlabel("Number of Mentions")
    plt.ylabel(f"Count {('','(logarithmic)')[log]}")

    plt.hist(x=trump_features['no_mentions'], bins=range(50), color='#FF00007F', log=log, label='Trump')
    plt.hist(x=chump_features['no_mentions'], bins=range(50), color='#0000FF7F', log=log, label='Chump')

    plt.legend()


def bar_heights(dataset, columns):
    return list(map(lambda column: dataset[column].mean(), columns))


def mean_punctuation_count_bar_chart():
    plt.title("Mean Punctuation Count Per Tweet")
    plt.xlabel("Punctuation Character")
    plt.ylabel("Mean Count of Punctuation (logarithmic)")

    width = 0.4

    trump_punctuation_columns = list(filter(lambda punctuation: punctuation in trump_features, string.punctuation))
    plt.bar(x=np.arange(len(trump_punctuation_columns)) - width/2, width=width, height=bar_heights(trump_features, trump_punctuation_columns), color='red', log=True, label='Trump')

    chump_punctuation_columns = list(filter(lambda punctuation: punctuation in chump_features, string.punctuation))
    plt.bar(x=np.arange(len(chump_punctuation_columns)) + width/2, width=width, height=bar_heights(chump_features, chump_punctuation_columns),tick_label=chump_punctuation_columns, color='blue', log=True, label='Chump')

    plt.legend()

# requires retweet_count & favourite_count to not be dropped in feature extraction
def mean_favourite_and_retweet_count_bar_chart():
    plt.title("Mean Favourite Count and Retweet Count Per Tweet")
    plt.ylabel("Mean Count")

    width = 0.4
    column_labels = ['favorite_count', 'retweet_count']

    plt.bar(x=np.arange(len(column_labels)) - width/2, width=width, height=bar_heights(trump_features, column_labels), color='red', label='Trump')
    plt.bar(x=np.arange(len(column_labels)) + width/2, width=width, height=bar_heights(chump_features, column_labels),tick_label=column_labels, color='blue', label='Chump')

    plt.legend()


# mean_punctuation_count_bar_chart()
# number_of_mentions_histogram()
# number_of_mentions_histogram(True)
# number_of_punctuation_characters_boxplots()
# number_of_hashtags_histogram()
# average_word_length_boxplots()
# tweet_length_boxplots()
mean_favourite_and_retweet_count_bar_chart()

fig = plt.gcf()
fig.set_size_inches(8, 6)
# fig.savefig('test2png.png', dpi=100)
plt.show()

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
