import re
import collections
import string

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


def create_text_without_mentions(dataset):
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


def extract_features(dataset, drop_length=True, drop_retweets_and_favorites=True):
    # get all tweet sources as a list
    tweet_sources = list(dataset['source'].unique())
    # get the index of the tweet source for each tweet
    dataset = dataset.assign(source=dataset['source'].apply(lambda x: tweet_sources.index(x)))

    # get the number of hashtags used in each tweet
    dataset = dataset.assign(no_hashtags=dataset['hashtags'].apply(len))

    # get the number of user mentions used in each tweet
    dataset = dataset.assign(no_mentions=dataset['user_mentions'].apply(len))

    # get the number of media items used in each tweet
    dataset = dataset.assign(no_media=dataset['media'].apply(media_length))

    # get the number of urls contained in each tweet
    dataset = dataset.assign(no_urls=dataset['text_without_mentions'].apply(count_urls))

    # remove urls from the text content of each tweet
    dataset = dataset.assign(text_without_mentions=dataset['text_without_mentions'].apply(remove_urls))

    # get the average length of the words in each tweet
    dataset = dataset.assign(avg_word_length=dataset['text_without_mentions'].apply(average_word_length))

    # get the number of punctuation used in each tweet
    dataset = dataset.assign(no_punctuation=dataset['text_without_mentions'].apply(count_punctuation))

    # get the length of each tweet
    dataset = dataset.assign(length=dataset['text_without_mentions'].apply(len))

    columns_to_drop = ['text', 'text_without_mentions', 'user_mentions', 'media',
                       'hashtags', 'symbols']

    if drop_length:
        columns_to_drop.append('length')

    if drop_retweets_and_favorites:
        columns_to_drop.append('retweet_count')
        columns_to_drop.append('favorite_count')

    return dataset.drop(columns=columns_to_drop)


def extract_text_features(X_train, X_test, include_urls=True):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

    if not include_urls:
        X_train = X_train.assign(text=X_train['text'].apply(remove_urls))
        X_test = X_test.assign(text=X_test['text'].apply(remove_urls))

    print("Setting stopwords")
    stop_words = set(stopwords.words('english'))

    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, articles):
            return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t not in stop_words]

    print("Creating count vectorizer")
    count_vect = CountVectorizer(tokenizer=LemmaTokenizer(),
                                 strip_accents='unicode',
                                 lowercase=True)
    print("Fitting count vectorizer")
    counts = count_vect.fit_transform(X_train['text'])

    print("Calculating term frequencies")
    tfidf_transformer = TfidfTransformer(use_idf=False)

    X_train_tf = tfidf_transformer.fit_transform(counts)
    X_test_tf = tfidf_transformer.transform(count_vect.transform(X_test['text']))

    return X_train_tf, X_test_tf
