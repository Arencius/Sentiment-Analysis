import numpy as np
from pywsd import lemmatize_sentence
from src.network.preprocessing import word2token
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from typing import List


def predict_sentiment(model: Sequential, text: str, sentiments: List, max_length: int, vocab: dict):
    """
    Given the text, neural network predicts the sentiment
    :param model:
    :param text:
    :param sentiments:
    :param max_length:
    :return:
    """
    tokenized = [[word2token(vocab, word) for word in lemmatize_sentence(text.lower())]]
    preprocessed_text = pad_sequences(tokenized, maxlen=max_length, padding='post')

    prediction = model.predict(preprocessed_text)[0]
    sentiment, confidence = sentiments[np.argmax(prediction)], max(prediction) * 100

    return 'Predicted: {}, with confidence: {:.2f}%'.format(sentiment, confidence)
