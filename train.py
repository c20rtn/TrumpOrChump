import pandas as pd
from feature_extraction import extract_features, extract_text_features, create_column_with_text_without_mentions
from sklearn.model_selection import train_test_split
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# LOAD DATA
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

accuracies = []
cms = []

for i in range(5):
    # split dataset into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(dataset[['favorite_count', 'is_quote_status', 'retweet_count', 'source', 'text', 'hashtags', 'symbols', 'user_mentions', 'media']], dataset['label'], test_size=0.20)
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)


    # REMOVE MENTIONS FROM TEXT
    X_train = create_column_with_text_without_mentions(X_train)
    X_test = create_column_with_text_without_mentions(X_test)


    # TEXT EXTRACTION
    X_train_tf, X_test_tf = extract_text_features(X_train, X_test, 'text')
    # X_train_tf, X_test_tf = feature_extraction.extract_text_features(X_train, X_test, 'text_without_mentions')


    # EXTRACT FEATURES
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)


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
    from models import logistic_regression, naive_bayes, svm, mlp

    # model = logistic_regression(joined_train, y_train)
    model = naive_bayes(X_train, y_train)
    # model = svm(joined_train, y_train)
    # model = mlp(joined_train, y_train)

    y_pred = model.predict(X_test)


    # ACCURACY MEASURES
    from sklearn.metrics import confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    # print(accuracy)
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)
    # print(cm)

print(cms)
print(accuracies)
print(np.average(accuracies))

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
