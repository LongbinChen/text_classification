import argparse
import numpy as np

def run(params):

    print('merging results files to make summary')
    names = params.dataname.split(",")
    with open(params.summary, 'w') as f:
        for i, idx in enumerate(params.results):
           f.write("\n=== data file %s ===\n" % names[i] )
           with open(idx, "r") as fin:
               cnt = 0
               for ln in fin:
                 f.write(ln)
                 cnt +=1
                 if cnt > params.maxline: break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+',  help='the resulting files, input')
    parser.add_argument('--summary', help='the summary file, output')
    parser.add_argument('--dataname', type=str, help='the name for each file, output')
    parser.add_argument('--maxline', type=int, default=10, help='the maximum number of line')

    params  = parser.parse_args()
    run(params)
