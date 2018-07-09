import argparse
import numpy as np, pandas as pd

def run(params):
    print('merging results files to make new submission')

    #load data
    input_data = []
    for s in params.input_submission:
       p_data = pd.read_csv(s)
       input_data.append(p_data)

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    all_cols = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    p_res = pd.DataFrame(np.zeros(input_data[0].shape), columns = all_cols)
    p_res['id'] = input_data[0]['id'] 
    print len(input_data)
    for i, w in enumerate(params.weight.split(",")): 
      w = float(w)
      p_res[label_cols] = p_res[label_cols] + input_data[i][label_cols]*w

    p_res.to_csv(params.submission, index=False)       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', help='the weight for each submssion, input, comma separated value')
    parser.add_argument('submission', help='the output submisison')
    parser.add_argument('input_submission', nargs='+',  help='the input submisison , input')

    params  = parser.parse_args()
    run(params)
