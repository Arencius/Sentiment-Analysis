import os
import logging
from keras.models import load_model
from gensim.models import Word2Vec
from src.gui.gui_window import App

if __name__ == '__main__':
    # TODO fix this
    logging.disable(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    MAX_LENGTH = 50

    MODEL = load_model('src/network/model.h5')
    VOCAB = Word2Vec.load('src/network/word2vec.model').wv.vocab

    App(MODEL, VOCAB, MAX_LENGTH)
