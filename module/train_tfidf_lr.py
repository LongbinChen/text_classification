import argparse
import os

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import utils
import data_utils
import text_utils

def run(params):

    print('Loading training data...')
    x_raw, y, _, _ = data_utils.load_data(params.train_data,
                                          params.category_label,
                                          params.limit_per_category,
                                          params.min_word_count)

    print('Preprocessing...')
    stopwords = text_utils.get_stopwords(params.stopwords)
    x_prep = [text_utils.preprocess(text, stopwords, params.keep_url) for text in x_raw]

    print('Grid searching for parameters...')
    ppl = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SGDClassifier())])
    parameters = {
        'tfidf__ngram_range': list(map(lambda x: (1, x), range(1, params.ngram_max_n + 1))),
        'tfidf__max_features': [5000],
        'clf__max_iter': [params.num_epochs],
        'clf__tol': list(map(float, params.tol.split(','))),
        'clf__loss': ['log'],
        'clf__penalty': ['l2'],
        'clf__alpha': list(map(float, params.alpha.split(','))),
        'clf__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(ppl, parameters, scoring='f1_weighted',  cv=10, refit=False, n_jobs=-1, verbose=1)
    grid_search.fit(x_prep, y)

    cv_scores = grid_search.cv_results_['mean_test_score']
    cv_stds = grid_search.cv_results_['std_test_score']
    cv_params = grid_search.cv_results_['params']

    for i, s in enumerate(cv_scores):
        print('F1 score is {} for parameters {}'.format(s, cv_params[i]))

    # return True if model with params p1 is simpler than p2
    def is_simpler_model(p1, p2):
        transform = lambda x: [x['tfidf__ngram_range'], -x['clf__alpha'], -x['clf__tol']]
        return transform(p1) < transform(p2)

    max_score = max(cv_scores)
    max_idx = np.argmax(cv_scores)
    score_threshold = max_score - cv_stds[max_idx]
    print('Max score is {}'.format(max_score))

    # select optimal model by one-standard-error rule
    best_param = cv_params[max_idx]
    best_score = max_score
    for i, s in enumerate(cv_scores):
        if s >= score_threshold and is_simpler_model(cv_params[i], best_param):
           best_param = cv_params[i]
           best_score = s

    print('Best score selected by 10-fold cross validation and one-standard-error rule: {}'.format(best_score))
    print('Best parameters selected by 10-fold cross validation and one-standard-error rule:')
    for name in parameters.keys():
        print('{}: {}'.format(name, best_param[name]))

    # train model with best params and whole training set
    tfidf = TfidfVectorizer(ngram_range=best_param['tfidf__ngram_range'],
                            max_features=best_param['tfidf__max_features'])
    x_tfidf = tfidf.fit_transform(x_prep)

    os.system('mkdir -p {}'.format(params.result))
    tfidf_dir = os.path.join(params.result, 'tfidf')
    os.system('touch {}'.format(tfidf_dir))
    joblib.dump(tfidf, tfidf_dir)

    del x_raw, x_prep, tfidf

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_train = x_tfidf[shuffle_indices]
    y_train = np.array(y, dtype=np.int32)
    y_train = y_train[shuffle_indices]

    print('Fitting...')

    clf = SGDClassifier(max_iter=best_param['clf__max_iter'],
                        tol=best_param['clf__tol'],
                        loss=best_param['clf__loss'],
                        penalty=best_param['clf__penalty'],
                        alpha=best_param['clf__alpha'],
                        class_weight=best_param['clf__class_weight'])
    clf = clf.fit(x_train, y_train)

    model_dir = os.path.join(params.result, 'clf')
    os.system('touch {}'.format(model_dir))
    joblib.dump(clf, model_dir)

    print('Classifying...')
    result = clf.predict(x_train)
    accuracy = 1.0 * (result == y_train).sum() / x_train.shape[0]
    print('Training accuracy: {}, Total: {}'.format(accuracy, x_train.shape[0]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--limit_per_category', default=0, type=int, help='limit per category for downsampling')
    parser.add_argument('--min_word_count', default=10, type=int, help='min word count for sample filtering')
    parser.add_argument('--keep_url', default=False, type=utils.str2bool, help='whether to keep url in text')
    parser.add_argument('--ngram_max_n', default=1, type=int, help='ngram max n for tfidf')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--tol', default='1e-3', type=str, help='tolerance for sgdclassifier convergence')
    parser.add_argument('--alpha', default='1e-5', type=str, help='penalty weight for sgdclassifier')
    parser.add_argument('stopwords', help='stopwords file name')
    parser.add_argument('category_label', help='original test label file name')
    parser.add_argument('train_data', help='original train data file name')
    parser.add_argument('result', help='a directory containing the trained model')

    params  = parser.parse_args()
    run(params)
