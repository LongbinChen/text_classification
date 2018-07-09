import argparse
import numpy as np 
import pandas as pd 
from subprocess import check_output

import tensorflow as tf
from keras import backend as K

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score

def get_model(params):
    inp = Input(shape=(params.maxlen, ))
    x = Embedding(params.max_features, params.embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def run(params):
    max_features = params.max_features
    maxlen = params.maxlen
    train = pd.read_csv(params.train_data)
    test = pd.read_csv(params.test_data)
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("CVxTz").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("CVxTz").values


    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    model = get_model(params)
    batch_size = params.batch_size
    epochs = params.num_epochs
    file_path="weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


    callbacks_list = [checkpoint, early] #early
    model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
    model.load_weights(file_path)

    y_test = model.predict(X_te)
    

    sample_submission = pd.read_csv(params.sample_submission)
    sample_submission[list_classes] = y_test
    sample_submission.to_csv(params.submission, index=False)
    for col in list_classes:
        gr = test[col].values
        pr = sample_submission[col].values
        auc = roc_auc_score(gr, pr)
        print col, auc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--embed_size', default=128, type=int, help='embed size, default 128')
    parser.add_argument( '--num_epochs', default=2, type=int, help='number of epochs')
    parser.add_argument( '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument( '--maxlen', default=100, type=int, help='max number of text ')
    parser.add_argument( '--max_features', default=20000, type=int, help='max number of input tokens')

    parser.add_argument('train_data', help='original train data file name')
    parser.add_argument('test_data', help='original test data file name')
    parser.add_argument('sample_submission', help='original sample_submission file name')
    parser.add_argument('submission', help='the output file containing the trained model')

    params  = parser.parse_args()
    run(params)
