import pandas as pd
import feature_extraction
from sklearn.model_selection import train_test_split
import data_cleansing

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# LOAD DATA
# read in individual datasets
trump_created = data_cleansing.cleanse_data_with_created_at(pd.read_json("Datasets/Full/trump_tweets_full.json"))
trump_tweets = data_cleansing.cleanse_data(trump_created[trump_created['created_at'] > '2017-11-07 00:00:00'])
general_tweets = pd.read_json("Datasets/general_tweets.json").sample(n=len(trump_tweets))

# label the datasets
trump_tweets['label'] = 1
general_tweets['label'] = 0

# join the datasets
dataset = pd.DataFrame()
dataset = dataset.append(trump_tweets)
dataset = dataset.append(general_tweets)



# split dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(dataset[['favorite_count', 'is_quote_status', 'retweet_count', 'source', 'text', 'hashtags', 'symbols', 'user_mentions', 'media']], dataset['label'], test_size=0.20)
X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)


# TEXT EXTRACTION
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 

nltk.download('punkt')
nltk.download('wordnet')

# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# preprocess, tokenize and filter stopwords and produce bag of words from tweet text
# produces sparse matrix, where each row represents a tweet and the given tweets word occurrences

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

count_vect = CountVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True)
counts = count_vect.fit_transform(X_train['text'])

print("Count Vectorizer Size : ", len(count_vect.get_feature_names()))

# divides occurrences by number of words in tweet
tfidf_transformer = TfidfTransformer(use_idf=False)

X_train_tf = tfidf_transformer.fit_transform(counts)
X_test_tf = tfidf_transformer.transform(count_vect.transform(X_test['text']))


# EXTRACT FEATURES
X_train = feature_extraction.extract_features(X_train)
X_test = feature_extraction.extract_features(X_test)


# DATA SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# JOIN SCALED FEATURES AND TEXT MEASURES
from scipy import sparse

joined_train = sparse.hstack([X_train_tf, sparse.csr_matrix(X_train)])
joined_test = sparse.hstack([X_test_tf, sparse.csr_matrix(X_test)])


# RUN TRAINING
import models

model = models.logistic_regression(joined_train, y_train)
# model = models.naive_bayes(X_train, y_train)
# model = models.svm(joined_train, y_train)
# model = models.mlp(joined_train, y_train)

y_pred = model.predict(joined_test)


# ACCURACY MEASURES
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy : ", accuracy, "\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)


pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
