import pandas as pd
import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


def average_word_length(sentence):
    words = regex.findall(sentence)
    return sum(len(word) for word in words) / len(words)

def media_length(x):
    if x is None:
        return 0.0
    else:
        return len(x)

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

# split dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(dataset[['favorite_count', 'is_quote_status', 'retweet_count', 'source', 'text', 'hashtags', 'symbols', 'user_mentions', 'media']], dataset['label'], test_size=0.25)
X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)

# # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#
# # preprocess, tokenize and filter stopwords and produce bag of words from tweet text
# # produces sparse matrix, where each row represents a tweet and the given tweets word occurences
# count_vect = CountVectorizer()
# counts = count_vect.fit_transform(X_train['text'])
#
# # divides occurences by number of words in tweet
# tfidf_transformer = TfidfTransformer(use_idf=False)
# tf = tfidf_transformer.fit_transform(counts)
#
# # add column, using row 0 to initialize datatype
# X_train['Term Frequencies'] = tf[0]
#
# # add data to column
# for i, row in X_train.iterrows():
#     X_train.at[i, 'Term Frequencies'] = tf[i]

X_train = feature_extraction.extract_features(X_train)

print(X_train.head(5))

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')


