import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Embedding, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from src.network.preprocessing import get_raw_dataset, prepare_dataset, word2token

# predefined constants
MAX_LENGTH = 50
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.00075
OPTIMIZER = Adam(LEARNING_RATE)

CALLBACKS = [EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights=True),
             ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)]

# loading and preprocessing the dataset
dataset = get_raw_dataset()
sentences, labels = prepare_dataset(dataset)

# creating embedding model
word2vec_model = Word2Vec(sentences, size=MAX_LENGTH, window=5, min_count=4, workers=4)
word2vec_model.save('word2vec.model')

# variables obtained with word2vec embedding model
VOCAB = word2vec_model.wv.vocab
WEIGHTS = word2vec_model.wv.vectors         # embedding matrix
VOCABULARY_SIZE, EMBEDDING_SIZE = WEIGHTS.shape

# tokenization of the sentences
sentences = [[word2token(word, VOCAB) for word in sentence] for sentence in sentences]

# padding the sentences with zeros at the end so every sentence has the same length
sentences = pad_sequences(sentences, maxlen=MAX_LENGTH, padding='post')

# weights to balance the classes
class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(labels), labels)))

# splitting the dataset
tweets_train, tweets_test, labels_train, labels_test = train_test_split(sentences, labels,
                                                                        test_size=0.3, stratify=labels)
tweets_valid, tweets_test, labels_valid, labels_test = train_test_split(tweets_test, labels_test,
                                                                        test_size=0.5, stratify=labels_test)

# building the model
CLASS_NUM = len(set(labels_train))

model = Sequential([Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE,
                              weights=[WEIGHTS],
                              input_length=MAX_LENGTH),
                    Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.15)),
                    Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2)),
                    GlobalAveragePooling1D(),
                    Dense(units=CLASS_NUM, activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=OPTIMIZER)

history = model.fit(tweets_train, labels_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(tweets_valid, labels_valid),
                    callbacks=CALLBACKS,
                    class_weight=class_weights)
