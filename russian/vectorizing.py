import gensim
import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing

""" USAGE: python vectorizing.py path/to/dataset.tsv path/to/vectorized method
path/to/dataset.tsv: dataset to model vectors, assumes the last column has prediction. default is lexical_entailment/russian/lrwc_vectorized.tsv
path/to/vectorized: desired output location
method: diff, asym

Example: python vectorizing.py default lexical_entailment/russian/lrwc_vectorized_diff.tsv diff
"""
if len(sys.argv) != 4:
    raise ValueError('Usage: python vectorizing.py path/to/model.bin path/to/dataset.tsv path/to/vectorized method')

path_to_dataset = '../lexical_entailment/russian/lrwc_vectorized.tsv' if sys.argv[1] == 'default' else sys.argv[1]
path_to_vectorized = sys.argv[2]
method = sys.argv[3]

df = pd.read_csv(path_to_dataset, sep='\t', header=None)

vectors_0 = df.iloc[:, :300]
vectors_1 = df.iloc[:, 300:600]
y = df.iloc[:, -1]

def merge_vectors(v1, v2, method):
    print "method", method
    if method == 'concat':
        return pd.concat([v1, v2, y], axis = 1)
    elif method == 'diff':
        # Roller 2014 says to normalize the difference
        diff = v1.values - v2.values
        print diff.shape
        return pd.concat([pd.DataFrame(diff), y], axis = 1)
    elif method == 'asym':
        # diff
        a = v1.values - v2.values
        # squared diff
        b = pd.DataFrame(np.sqrt(np.square(a).sum(axis=1)))
        return pd.concat([pd.DataFrame(a), pd.DataFrame(b), y], axis = 1)

vectors_x = merge_vectors(vectors_0, vectors_1, method)
print vectors_x.shape
vectors_x.to_csv(path_to_vectorized, sep='\t', header=False, index=False)
