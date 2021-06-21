import os
import logging
from keras.models import load_model
from gensim.models import Word2Vec
from src.gui.gui_window import App


if __name__ == '__main__':
    MAX_LENGTH = 50

    # TODO fix this
    logging.disable(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # deserialize pre-trained models
    model = load_model('src/network/model.h5')
    vocab = Word2Vec.load('src/network/word2vec.model').wv.vocab

    # run the application
    App(model, vocab, MAX_LENGTH)
