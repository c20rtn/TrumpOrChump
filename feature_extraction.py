import re
import collections
import string
import pandas as pd

words_regex = re.compile('[^\W]\S+[^\W]|\w\w|\w')
punctuation_regex = re.compile('[^\w #]')


def average_word_length(sentence):
    words = words_regex.findall(sentence)
    return sum(len(word) for word in words) / len(words)

def media_length(x):
    if x is None:
        return 0.0
    else:
        return len(x)

def create_column_with_text_without_mentions(dataset):
    return dataset.assign(text_without_mentions=dataset['text'].apply(remove_mentions))

def remove_mentions(text):
    return re.sub(pattern='@\w+', string=text, repl="")

def count_each_punctuation(text):
    counts = collections.Counter(text)
    val = {k: v for k, v in counts.items() if k in string.punctuation}
    # print(val)
    return val

def count_punctuation(text, emojis=True):
    if not emojis:
        text = text.encode('ascii', 'ignore').decode('ascii')
    punctuation = punctuation_regex.findall(text)
    return len(punctuation)

def extract_features(dataset, drop_length=True):
    # get all tweet sources as a list
    tweet_sources = list(dataset['source'].unique())
    # get the index of the tweet source from the list and store it in the 'source' column
    dataset = dataset.assign(source=dataset['source'].apply(lambda x: tweet_sources.index(x)))

    # get the number of hashtags used in the tweet and store it in the 'no_hashtags' column
    dataset = dataset.assign(no_hashtags=dataset['hashtags'].apply(len))

    # get the number of user mentions used in the tweet and store it in the 'no_mentions' column
    dataset = dataset.assign(no_mentions=dataset['user_mentions'].apply(len))

    # get the number of media items used in the tweet and store it in the 'no_media' column
    dataset = dataset.assign(no_media=dataset['media'].apply(media_length))

    text_column = 'text'

    if 'text_without_mentions' in dataset.columns:
        text_column = 'text_without_mentions'

    # get the average length of the words in the tweet and store it in the 'avg_word_length' column
    dataset = dataset.assign(avg_word_length=dataset[text_column].apply(average_word_length))

    # get the number of punctuation in the text
    dataset = dataset.assign(no_punctuation=dataset[text_column].apply(count_punctuation))

    # get the length of each tweet and store it in the 'length' column
    dataset = dataset.assign(length=dataset[text_column].apply(len))

    # get the count of each punctuation character and store it in a separate column per character
    dataset = dataset.join(dataset[text_column].apply(count_each_punctuation).apply(pd.Series).fillna(0))

    # columns that should be kept [['is_quote_status', 'source', 'no_hashtags', 'no_mentions', 'no_media', 'avg_word_length', 'no_punctuation']]
    columns_to_drop=['favorite_count', 'retweet_count', 'text', 'text_without_mentions', 'user_mentions', 'media', 'hashtags', 'symbols']

    if drop_length:
        columns_to_drop.append('length')

    return dataset.drop(columns=columns_to_drop)


def extract_text_features(X_train, X_test, column_name):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import nltk
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

    # preprocess, tokenize and filter stopwords and produce bag of words from tweet text
    # produces sparse matrix, where each row represents a tweet and the given tweets word occurrences
    
    print("Setting stopwords")
    stopWords = set(stopwords.words('english'))
    
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, articles):
            return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t not in stopWords]

    print("Count Vectorizer")
    count_vect = CountVectorizer(
                                tokenizer=LemmaTokenizer(),
                                strip_accents='unicode',
                                #stop_words='english',
                                lowercase=True
    )
    counts = count_vect.fit_transform(X_train[column_name])

    # divides occurrences by number of words in tweet
    tfidf_transformer = TfidfTransformer(use_idf=False)

    X_train_tf = tfidf_transformer.fit_transform(counts)
    X_test_tf = tfidf_transformer.transform(count_vect.transform(X_test[column_name]))

    return X_train_tf, X_test_tf
