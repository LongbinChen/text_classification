import argparse
import numpy as np, pandas as pd

def run(params):
    print('split the data in to train and validation')
    train = pd.read_csv(params.train_data)
    msk = np.random.rand(len(train)) < params.weight
    train_split = train[msk]
    val_split = train[~msk]
    sample_sub_cols = ["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    all_sub_cols = ["id","comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    #train_split = train_split[all_sub_cols]
    train_split.to_csv(params.train_split, index = False)
    #val_split = val_split[all_sub_cols]
    val_split.to_csv(params.val_split, index = False)
    sample_sub = val_split[sample_sub_cols] 
    sample_sub.to_csv(params.val_sample_submission, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=float, default=0.8, help='the weight for train in split, default 0.8')
    parser.add_argument('train_data', help='the input train data')
    parser.add_argument('train_split', help='the output train')
    parser.add_argument('val_split', help='the output validation ')
    parser.add_argument('val_sample_submission', help='the output sample submission')

    params  = parser.parse_args()
    run(params)
