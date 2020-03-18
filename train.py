import pandas as pd
import feature_extraction
from sklearn.model_selection import train_test_split


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# read in individual datasets
trump_tweets = pd.read_json("Datasets/trump_tweets.json")
general_tweets = pd.read_json("Datasets/general_tweets.json")

# print(trump_tweets['favorite_count'].mean())
# print(trump_tweets['retweet_count'].mean())
# print(general_tweets['favorite_count'].mean())
# print(general_tweets['retweet_count'].mean())
# print(feature_extraction.extract_features(general_tweets)['no_media'].mean())
# print(feature_extraction.extract_features(trump_tweets)['no_media'].mean())

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

# X_train = feature_extraction.extract_features(X_train)
# X_test = feature_extraction.extract_features(X_test)

# DATA SCALING
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# TEXT EXTRACTION

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# preprocess, tokenize and filter stopwords and produce bag of words from tweet text
# produces sparse matrix, where each row represents a tweet and the given tweets word occurences
count_vect = CountVectorizer()
counts = count_vect.fit_transform(X_train['text'])

# divides occurences by number of words in tweet
tfidf_transformer = TfidfTransformer(use_idf=False)
tf = tfidf_transformer.fit_transform(counts)
# print(X_train.to_numpy())

from scipy import sparse
# print(sparse.csr_matrix(feature_extraction.extract_features(X_train).values))
joined_train = sparse.hstack([tf, sparse.csr_matrix(feature_extraction.extract_features(X_train).values)])

X_test_tf = tfidf_transformer.transform(count_vect.transform(X_test['text']))

joined_test = sparse.hstack([X_test_tf, sparse.csr_matrix(feature_extraction.extract_features(X_test).values)])

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(random_state=0, max_iter=5000)
# lr.fit(joined_train, y_train)
# y_pred = lr.predict(joined_test)


# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

# nb = GaussianNB()
# nb.fit(X_train, y_train)
# y_pred = nb.predict(X_test)


# SVM
from sklearn.svm import LinearSVC

svc = LinearSVC(random_state=0, tol=1e-03, max_iter=10000)
svc.fit(joined_train, y_train)
y_pred = svc.predict(joined_test)


# MLP
from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(random_state=0)
# mlp.fit(X_train, y_train)
# y_pred = mlp.predict(X_test)

# ACCURACY MEASURES
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)


pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')


