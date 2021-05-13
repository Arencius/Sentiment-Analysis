import numpy as np
from pywsd import lemmatize_sentence
from src.network.preprocessing import word2token
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from typing import List


def predict_sentiment(model: Sequential, text: str, sentiments: List, max_length: int, vocab: dict):
    """
    Function predicts the sentiment of the given text using the pretrained neural network
    :param model: trained model
    :param text: text to predict the sentiment
    :param sentiments:
    :param max_length: max length of the sentence
    :param vocab: vocabulary of all learnt words
    :return: predicted sentiment with the confidence
    """
    tokenized = [[word2token(vocab, word) for word in lemmatize_sentence(text.lower())]]
    preprocessed_text = pad_sequences(tokenized, maxlen=max_length, padding='post')

    prediction = model.predict(preprocessed_text)[0]
    sentiment, confidence = sentiments[np.argmax(prediction)], max(prediction) * 100

    return 'Predicted: {}, with confidence: {:.2f}%'.format(sentiment, confidence)
