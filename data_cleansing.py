import re


def cleanse_data(dataset):
    # remove all non-english tweets
    if 'lang' in dataset.columns:
        dataset = dataset[dataset.lang == "en"]
    # remove all tweets withheld for copyright purposes
    if 'withheld_copyright' in dataset.columns:
        dataset = dataset[dataset.withheld_copyright != 1]
    # remove deleted tweets
    if 'delete' in dataset.columns:
        dataset = dataset[dataset.delete.isna()]

    filtered_dataset = dataset[['favorite_count', 'is_quote_status', 'retweet_count', 'source', 'text', 'hashtags',
                                'symbols', 'user_mentions', 'media']]

    return filtered_dataset


def cleanse_data_with_created_at(dataset):
    # remove all non-english tweets
    if 'lang' in dataset.columns:
        dataset = dataset[dataset.lang == "en"]
    # remove all tweets withheld for copyright purposes
    if 'withheld_copyright' in dataset.columns:
        dataset = dataset[dataset.withheld_copyright != 1]
    # remove deleted tweets
    if 'delete' in dataset.columns:
        dataset = dataset[dataset.delete.isna()]

    filtered_dataset = dataset[['favorite_count', 'is_quote_status', 'retweet_count', 'source', 'text', 'hashtags',
                                'symbols', 'user_mentions', 'media', 'created_at']]

    return filtered_dataset

# general
#       ['created_at', 'id', 'id_str', 'text', 'source', 'truncated', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
#        'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
#        'contributors', 'retweeted_status', 'is_quote_status', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count',
#        'entities', 'extended_entities', 'favorited', 'retweeted', 'possibly_sensitive', 'filter_level', 'lang', 'timestamp_ms',
#        'display_text_range', 'quoted_status_id', 'quoted_status_id_str', 'quoted_status', 'quoted_status_permalink',
#        'extended_tweet', 'delete', 0, 'hashtags', 'media', 'symbols', 'urls', 'user_mentions', 'withheld_in_countries']
# trump
#       ['contributors', 'coordinates', 'created_at', 'entities', 'favorite_count', 'favorited', 'geo', 'id', 'id_str',
#        'in_reply_to_screen_name', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
#        'in_reply_to_user_id_str', 'is_quote_status', 'lang', 'place', 'retrieved_utc', 'retweet_count', 'retweeted',
#        'source', 'text', 'truncated', 'user', 'possibly_sensitive', 'extended_entities', 'quoted_status', 'quoted_status_id',
#        'quoted_status_id_str', 'retweeted_status', 'withheld_copyright', 'withheld_in_countries', 'withheld_scope', 'scopes',
#        'hashtags', 'symbols', 'urls', 'user_mentions', 'media']


def source_regex(text):
    if text != text:
        return text

    regex = re.compile('<.*>(.*)<.*>')
    results = regex.findall(text)
    if len(results) == 0:
        return text

    return results[0]
