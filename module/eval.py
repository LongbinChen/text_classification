import argparse
import numpy as np, pandas as pd
import math
from sklearn.metrics import roc_auc_score

def run(params):
    print('comparing the prediction with ground truth')
    ground_truth = pd.read_csv(params.ground_truth)
    pred = pd.read_csv(params.submission)

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    n = ground_truth.shape[0]
    print n 
    sum = 0.0
    with open(params.result, 'w') as fout:
      for col in label_cols: 
        gr = ground_truth[col].values
        pr = pred[col].values
        auc = roc_auc_score(gr, pr)
        sum += auc
        print col, auc
        fout.write("%s\t%f\n" % (col, auc))
      print sum, sum / len(label_cols)
      fout.write("total\t%f\n" %  (sum / len(label_cols)))
    
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth', help='input, the ground truth')
    parser.add_argument('submission', help='input, the output prediction submisison')
    parser.add_argument('result', help='output, the summary')
    parser.add_argument('--se', help='parameters, the summary')
    return parser
    
if __name__ == '__main__':
    parser = create_parser()
    params  = parser.parse_args()
    run(params)
