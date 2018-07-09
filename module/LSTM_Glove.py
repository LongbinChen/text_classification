# -*- coding: utf-8 -*-
import argparse
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import keras.backend as K
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

def loss(y_true, y_pred):
     return K.binary_crossentropy(y_true, y_pred)
    

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))


def schedule(ind):
    a = [0.002,0.003, 0.000]
    return a[ind]

def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

stop = stopwords.words('english') + list(string.punctuation)
def del_stop_words(sent):
    sent = sent.decode("utf8")
    return " ".join([i for i in word_tokenize(sent.lower()) if i not in stop])
   
def remove_stop_words(list_sentense):
    return [del_stop_words(s) for s in list_sentense] 

def run(params):
    stop = stopwords.words('english')
    max_features = params.max_features
    maxlen = params.maxlen
    embed_size = params.embed_size

    train = pd.read_csv(params.train_data)
    test = pd.read_csv(params.test_data)
    list_sentences_train = train["comment_text"].fillna("_na_").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("_na_").values

    if params.remove_stopwords == "True":
       list_sentences_train = remove_stop_words(list_sentences_train)
       list_sentences_test = remove_stop_words(list_sentences_test)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(params.embed_file))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True,dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    #model.compile(loss=loss, optimizer='nadam', metrics=['accuracy'])
    model.compile(loss=params.loss, optimizer='nadam', metrics=['accuracy'])

    lr = callbacks.LearningRateScheduler(schedule)
    [X_train, X_val, y_train, y_val] = train_test_split(X_t, y, train_size=0.95)

    ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

    model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_val, y_val), callbacks=[lr, ra_val])

    y_test = model.predict([X_te], batch_size=1024, verbose=1)
    sample_submission = pd.read_csv(params.sample_submission)
    sample_submission[list_classes] = y_test
    sample_submission.to_csv(params.submission, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--remove_stopwords', default="False", type=str, help='remove stop words, default True')
    parser.add_argument( '--embed_size', default=50, type=int, help='embed size, default 50')
    parser.add_argument( '--loss', default='binary_crossentropy', type=str, help='loss function, default binary_crossentropy')
    parser.add_argument( '--num_epochs', default=2, type=int, help='number of epochs')
    parser.add_argument( '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument( '--maxlen', default=100, type=int, help='max number of text ')
    parser.add_argument( '--max_features', default=20000, type=int, help='max number of input tokens')

    parser.add_argument('train_data', help='original train data file name')
    parser.add_argument('test_data', help='original test data file name')
    parser.add_argument('embed_file', help='embed file')
    parser.add_argument('sample_submission', help='original sample_submission file name')
    parser.add_argument('submission', help='the output file containing the trained model')

    params  = parser.parse_args()
    run(params)
