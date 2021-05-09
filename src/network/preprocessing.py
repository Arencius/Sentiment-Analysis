import re
import pandas as pd


def load_sentiment_data(path: str, value: str) -> pd.DataFrame:
    """
    TODO: Function description
    :param path: path to the file
    :param value: sentiment of the tweets in given file (all sentences in the file belong to only one class)
    :return: dataframe with tweets and corresponding sentiments
    """
    with open(path) as file:
        tweets = file.read().split(',')

    data_length = len(tweets)
    sentiments = [value] * data_length

    return pd.DataFrame({
        'text': tweets,
        'sentiment': sentiments
    })


def preprocess_tweet(tweet: str) -> str:
    """
    Function preprocesses the single tweet using regular expression, by removing:
    - user tags, for example: @username,
    - all hashtags, for example: #good,
    - punctuation, as it's irrelevant in model training process,
    - new line markups existing in one of the files,
    - digits and numerals such as 1st, 2th etc.
    :param tweet: tweet to be processed
    :return: cleaned text
    """
    pattern = f'(@[A-Za-z0-9]*|' \
              f'#[A-Za-z0-9]*|' \
              f'[\.\?!,:\"&;()\n\t]|' \
              f'<br />|' \
              f'\d[a-z]*)'

    return re.sub(pattern, '', tweet).lower().strip()
