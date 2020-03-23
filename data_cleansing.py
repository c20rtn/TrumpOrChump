import re

source_regex = re.compile('<.*>(.*)<.*>')


def cleanse_data(dataset, include_created_at=False):
    # remove all non-english tweets
    if 'lang' in dataset.columns:
        dataset = dataset[dataset.lang == "en"]
    # remove all tweets withheld for copyright purposes
    if 'withheld_copyright' in dataset.columns:
        dataset = dataset[dataset.withheld_copyright != 1]
    # remove deleted tweets
    if 'delete' in dataset.columns:
        dataset = dataset[dataset.delete.isna()]

    columns_to_keep = ['favorite_count', 'is_quote_status', 'retweet_count', 'source', 'text', 'hashtags',
                       'symbols', 'user_mentions', 'media']

    if include_created_at:
        columns_to_keep.append('created_at')

    return dataset[columns_to_keep]


def extract_source(text):
    if text != text:
        return text

    results = source_regex.findall(text)
    if len(results) == 0:
        return text

    return results[0]
