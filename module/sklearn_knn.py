import argparse
import gzip
import numpy
import tensorflow as tf
import time
import sys
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import os
import glob
import pickle
from numpy import loadtxt

def refine_single_email(email):
    """
    Delete the unnecessary information in the header of emails
    Deletes only lines in the email that starts with 'Path:', 'Newsgroups:', 'Xref:'
    parameter is a string.
    returns a string.
    """

    parts = email.split('\n')
    newparts = []

    # finished is when we have reached a line with something like 'Lines:' at the begining of it
    # this is because we want to only remove stuff from headers of emails
    # look at the dataset!
    finished = False
    for part in parts:
        if finished:
            newparts.append(part)
            continue
        if not (part.startswith('Path:') or part.startswith('Newsgroups:') or part.startswith('Xref:')) and not finished:
            newparts.append(part)
        if part.startswith('Lines:'):
            finished = True

    return '\n'.join(newparts)

def refine_all_emails(file_data):
    """
    Does `refine_single_email` for every single email included in the list
    parameter is a list of strings
    returns NOTHING!
    """

    for i, email in zip(range(len(file_data)), file_data):
        file_data[i] = refine_single_email(email)


def bag_of_words(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)

def sklearn_knn(params):
  # load data
  # training data is a list of strings, each sample a line, train label is a label per line 
  train_data = [ln.strip("\n") for ln in open(params.train_data, "r").readlines()]
  train_label = loadtxt(params.train_label, comments="#", delimiter=",", unpack=True, dtype=int)
  test_data = [ln.strip("\n") for ln in open(params.test_data, "r").readlines()]
  test_label = loadtxt(params.test_label, comments="#", delimiter=",", unpack=True, dtype=int)

  print("training data size : %d. " % (len(train_data)))
  print("training label size : %d. " % (len(train_label)))
  print("test data size : %d. " % (len(test_data)))
  print("test label size : %d. " % (len(test_label)))
  all_data = train_data + test_data

  #calculate BOW representation
  word_counts = bag_of_words(all_data)
  
  #TFIDF
  tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
  all_X = tf_transformer.transform(word_counts)
  train_X = all_X[:len(train_data), :]
  test_X = all_X[len(train_data):, :]

  #KNN classifier
  
  n_neighbors =  params.neighbors
  weights = 'uniform'
  weights = 'distance'
  clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
  clf.fit(train_X, train_label)
  y_predicted = clf.predict(test_X)

  result = sklearn.metrics.classification_report(test_label, y_predicted)
  print result
  s = pickle.dumps(clf)
  with open(params.model, "w") as model_file:
    model_file.write(s)
  with open(params.result, "w") as result_file:
    result_file.write(result)

  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument( '--neighbors', default=11, type=int, help='num of neighboers in knn')
  '''
  parser.add_argument( '--num_epochs', default=2, type=int, help='number of epochs')
  parser.add_argument( '--batch_size', default=64, type=int, help='batch size')
  parser.add_argument( '--image_size', default=28, type=int, help='image size')
  parser.add_argument( '--pixel_depth', default=255, type=int, help='pixel_depth')
  parser.add_argument( '--eval_batch_size', default=64, type=int, help='eval batch size')
  parser.add_argument( '--num_channels', default=1, type=int, help='number of channels')
  parser.add_argument( '--num_img', default=60000, type=int, help='number of images ')
  parser.add_argument( '--test_num_img', default=10000, type=int, help='number of test images ')
  parser.add_argument( '--eval_frequency', default=100, type=int, help='eval frequency')
  '''
  parser.add_argument( 'train_data', help='original train data file name')
  parser.add_argument( 'train_label', help='original train label file name')
  parser.add_argument( 'test_data', help='original test data file name')
  parser.add_argument( 'test_label', help='original test label file name')
  parser.add_argument( 'model', help='model file')
  parser.add_argument( 'result', help='result file')

  params  = parser.parse_args()
  sklearn_knn(params)


