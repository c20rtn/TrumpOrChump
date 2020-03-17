import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

#preprocess, tokenize and filter stopwords and produce bag of words from tweet text
#produces sparse matrix, where each row represents a tweet and the given tweets word occurences
count_vect = CountVectorizer()
counts = count_vect.fit_transform(dataset['text'])

#divides occurences by number of words in tweet
tfidf_transformer = TfidfTransformer(use_idf=False)
tf = tfidf_transformer.fit_transform(counts)

#add column, using row 0 to initialize datatype
dataset['Term Frequencies'] = tf[0]

#add data to column
for i, row in dataset.iterrows():
    dataset.at[i, 'Term Frequencies'] = tf[i]

print(dataset.head(5))

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')


