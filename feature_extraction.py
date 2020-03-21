import re
import collections
import string
import pandas as pd

words_regex = re.compile('[^\W]\S+[^\W]|\w\w|\w')
punctuation_regex = re.compile('[^\w #]')
url_regex_string = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
url_regex = re.compile(url_regex_string)

def average_word_length(sentence):
    words = words_regex.findall(sentence)
    if len(words) == 0:
        return 0

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
    return {k: v for k, v in counts.items() if k in string.punctuation}

def count_punctuation(text, emojis=True):
    if not emojis:
        text = text.encode('ascii', 'ignore').decode('ascii')
    punctuation = punctuation_regex.findall(text)
    return len(punctuation)

def count_urls(text):
    urls = url_regex.findall(text)
    return len(urls)

def remove_urls(text):
    return re.sub(pattern=url_regex_string, string=text, repl="")

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

    dataset = dataset.assign(no_urls=dataset[text_column].apply(count_urls))

    if text_column is 'text':
        dataset = dataset.assign(text=dataset[text_column].apply(remove_urls))
    else:
        dataset = dataset.assign(text_without_mentions=dataset[text_column].apply(remove_urls))

    # get the average length of the words in the tweet and store it in the 'avg_word_length' column
    dataset = dataset.assign(avg_word_length=dataset[text_column].apply(average_word_length))

    # get the number of punctuation in the text
    dataset = dataset.assign(no_punctuation=dataset[text_column].apply(count_punctuation))

    # get the length of each tweet and store it in the 'length' column
    dataset = dataset.assign(length=dataset[text_column].apply(len))

    # get the count of each punctuation character and store it in a separate column per character
    # dataset = dataset.join(dataset[text_column].apply(count_each_punctuation).apply(pd.Series).fillna(0))

    # 94.0  (urls, no_urls, no no_pun, no pun_counts),      93.8    (no urls, no_urls, no no_pun, no pun_counts)
    # 91.0  (urls, no_urls, no no_pun, pun_counts),         92.1    (no urls, no_urls, no no_pun, pun_counts)
    # 94.2  (urls, no_urls, no_pun, no pun_counts),         94.1    (no urls, no_urls, no_pun, no pun_counts)       BEST, CHOSEN
    # 87.8  (urls, no_urls, no_pun, pun_counts),            92.2    (no urls, no_urls, no_pun, pun_counts)

    # 93.9  (urls, no no_urls, no no_pun, no pun_counts),   93.8    (no urls, no no_urls, no no_pun, no pun_counts)
    # 91.7  (urls, no no_urls, no no_pun, pun_counts),      92.7    (no urls, no no_urls, no no_pun, pun_counts)
    # 94.0  (urls, no no_urls, no_pun, no pun_counts),      94.0    (no urls, no no_urls, no_pun, no pun_counts)
    # 86.3  (urls, no no_urls, no_pun, pun_counts),         91.4    (no urls, no no_urls, no_pun, pun_counts)

    # columns that should be kept [['is_quote_status', 'source', 'no_hashtags', 'no_mentions', 'no_media', 'avg_word_length', 'no_punctuation']]
    columns_to_drop=['favorite_count', 'retweet_count', 'text', 'text_without_mentions', 'user_mentions', 'media', 'hashtags', 'symbols']

    if drop_length:
        columns_to_drop.append('length')

    return dataset.drop(columns=columns_to_drop)


def extract_text_features(X_train, X_test, column_name, include_urls = True):
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

    if not include_urls:
        if column_name is 'text':
            X_train = X_train.assign(text=X_train[column_name].apply(remove_urls))
            X_test = X_test.assign(text=X_test[column_name].apply(remove_urls))
        else:
            X_train = X_train.assign(text_without_mentions=X_train[column_name].apply(remove_urls))
            X_test = X_test.assign(text_without_mentions=X_test[column_name].apply(remove_urls))

    # 94.2  (urls, urls in counts, no_urls, no_pun, no pun_counts, text_mentions)
    # 94.1  (urls, no urls in counts, no_urls, no_pun, no pun_counts, text_mentions)
    # 94.3  (no urls, urls in counts, no_urls, no_pun, no pun_counts, text_mentions)        94.28
    # 94.3  (no urls, no urls in counts, no_urls, no_pun, no pun_counts, text_mentions)     94.34    BEST, CHOSEN
    # 93.6  (urls, urls in counts, no_urls, no_pun, no pun_counts, no_text_mentions)
    # 93.5  (urls, no urls in counts, no_urls, no_pun, no pun_counts, no_text_mentions)
    # 93.4  (no urls, urls in counts, no_urls, no_pun, no pun_counts, no_text_mentions)
    # 93.5  (no urls, no urls in counts, no_urls, no_pun, no pun_counts, no_text_mentions)

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
