import re

regex = re.compile('[^\W]\S+[^\W]|\w\w|\w')


def average_word_length(sentence):
    words = regex.findall(sentence)
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


def extract_features(dataset):
    # get all tweet sources as a list
    tweet_sources = list(dataset['source'].unique())
    # get the index of the tweet source from the list and store it in the 'source' column
    dataset = dataset.assign(source=dataset['source'].apply(lambda x: tweet_sources.index(x)))

    # get the length of each tweet and store it in the 'length' column
    dataset = dataset.assign(length=dataset['text'].apply(len))

    # get the number of hashtags used in the tweet and store it in the 'no_hashtags' column
    dataset = dataset.assign(no_hashtags=dataset['hashtags'].apply(len))

    # get the number of user mentions used in the tweet and store it in the 'no_mentions' column
    dataset = dataset.assign(no_mentions=dataset['user_mentions'].apply(len))

    # get the number of media items used in the tweet and store it in the 'no_media' column
    dataset = dataset.assign(no_media=dataset['media'].apply(media_length))

    # get the average length of the words in the tweet and store it in the 'avg_word_length' column
    dataset = dataset.assign(avg_word_length=dataset['text'].apply(average_word_length))

    # get the length of each tweet and thr average word length without mentions
    if 'text_without_mentions' in dataset.columns:
        dataset = dataset.assign(
            avg_word_length_without_mentions=dataset['text_without_mentions'].apply(average_word_length))

        dataset = dataset.assign(length_without_mentions=dataset['text_without_mentions'].apply(len))

        return dataset[['is_quote_status', 'source', 'no_hashtags', 'no_mentions', 'no_media'
            , 'avg_word_length_without_mentions', 'length_without_mentions']]

    return dataset[['is_quote_status', 'source', 'length', 'no_hashtags', 'no_mentions', 'no_media', 'avg_word_length']]


def extract_text_features(X_train, X_test, column_name):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import nltk
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # nltk.download('punkt')
    # nltk.download('wordnet')
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

    # preprocess, tokenize and filter stopwords and produce bag of words from tweet text
    # produces sparse matrix, where each row represents a tweet and the given tweets word occurrences
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, articles):
            return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

    count_vect = CountVectorizer(
                                 tokenizer=LemmaTokenizer(),
                                 strip_accents='unicode',
                                 stop_words='english',
                                 lowercase=True
    )
    counts = count_vect.fit_transform(X_train[column_name])

    # divides occurrences by number of words in tweet
    tfidf_transformer = TfidfTransformer(use_idf=False)

    X_train_tf = tfidf_transformer.fit_transform(counts)
    X_test_tf = tfidf_transformer.transform(count_vect.transform(X_test[column_name]))

    return X_train_tf, X_test_tf
