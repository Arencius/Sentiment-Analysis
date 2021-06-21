import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from pywsd.utils import lemmatize_sentence
from nltk.corpus import stopwords


def load_sentiment_data(path: str, value: str) -> pd.DataFrame:
    """
    Loads the processedNegative/Positive/Neutral.csv file and returns the ready-to-use dataframe
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
        'sentiment': sentiments})


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


def word2token(vocab: dict, word: str) -> int:
    """
    Transforms the word to integer based on its index in the vocabulary. Returns 0 if not in vocab
    :param vocab: vocabulary of the trained word2vec model
    :param word: word to transform to numerical form
    :return: Numerical form of the word, index in the vocabulary
    """
    return vocab.get(word).index if word in vocab.keys() else 0


def token2word(word2vec: Word2Vec, token: int) -> str:
    """
    Transforms the token to word on this index in the vocabulary
    :param word2vec: trained wor2vec model
    :param token: index of the word
    :return: word on the given index
    """
    return word2vec.wv.index2word[token]


def filter_stopwords(sentence: str) -> str:
    """
    Removes stop words from the given sentence
    :param sentence: raw sentence
    :return: sentence after removing the stop words
    """
    stop_words = stopwords.words('english')
    return ' '.join([word for word in sentence if not word in stop_words])


def get_raw_dataset() -> pd.DataFrame:
    """
    Loads and returns concatenated dataset from the ./data folder
    :return: unprocessed dataset
    """
    # Simple additional sentences to train the network the differences in understanding 'good' and 'bad'
    texts = ['not good not bad', 'not great not terrible', 'good and bad', 'could be better', 'could be worse',
             'I thought it would be terrible but it was good',
             'not bad not good', 'it was good', 'it was not good', 'it was bad', 'it was not bad', 'it was nice']
    sentiments = ['neutral', 'neutral', 'neutral', 'negative', 'positive', 'positive', 'neutral', 'positive',
                  'negative', 'negative', 'positive', 'positive']

    sentences = {'text': texts,
              'sentiment': sentiments}

    # loading additional data
    neutral_data = load_sentiment_data('src/data/processedNeutral.csv', value='neutral')
    positive_data = load_sentiment_data('src/data/processedPositive.csv', value='positive')
    negative_data = load_sentiment_data('src/data/processedNegative.csv', value='negative')

    # loading the movie reviews dataset
    imdb_data = pd.read_csv('src/data/IMDB Dataset.csv', names = ['text', 'sentiment'], header=0)

    # loading the twitter dataset
    tweets = pd.read_csv('src/data/Tweets.csv')
    tweets = tweets[['text', 'airline_sentiment']].rename(columns={'airline_sentiment': 'sentiment'})

    return shuffle(pd.concat([negative_data, neutral_data, positive_data, imdb_data, sentences, tweets]))


def prepare_dataset(data: pd.DataFrame):
    """
    Preprocesses the dataset by lemmatizing and padding the sentences.
    :param data: dataset
    :return: fully preprocessed dataset, split into sentences and labels
    """

    # cleaning the sentences, i.e. removing punctuation, hashtags etc.
    data['text'] = data['text'].apply(preprocess_tweet)

    # filtering and lemmatizing the sentences
    texts = np.array(data['text'])
    texts = [filter_stopwords(sentence) for sentence in texts]

    texts = [lemmatize_sentence(sentence) for sentence in texts]

    # encode labels to integer form
    encoder = LabelEncoder()

    labels = np.array(data['sentiment'])
    labels = encoder.fit_transform(labels)

    return texts, labels