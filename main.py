import os
import logging
from src.network.model_prediction import predict_sentiment
from keras.models import load_model
from gensim.models import Word2Vec

if __name__ == '__main__':
    # TODO fix this
    logging.disable(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    SENTIMENTS = ['Negative', 'Neutral', 'Positive']
    MAX_LENGTH = 50

    MODEL = load_model('src/network/model.h5')
    VOCAB = Word2Vec.load('src/network/word2vec.model').wv.vocab

    text = input('Your sentence: ')
    predicted = predict_sentiment(MODEL, text, SENTIMENTS, MAX_LENGTH, VOCAB)

    print(predicted)
