#!/usr/bin/env python
import gensim 
import pandas as pd
import numpy as np
import sys

""" USAGE: ./vectorizing.py path/to/model.bin path/to/dataset.tsv path/to/vectorized method
path/to/model.bin: path to the binary gensim model. If 'default' is passed in, uses Google's pre-trained word2vec model
path/to/dataset.tsv: dataset to vectorize. assumes the last column has prediction
method: right now just 'concat'
path/to/vectorized: desired output location

Example: ./vectorizing.py default lexical_entailment/bless2011/data_lex_train.tsv \
lexical_entailment/bless2011/data_lex_train_vectorized2.tsv \
concat
"""

if len(sys.argv) != 5:
	raise ValueError('Usage: ./vectorizing.py path/to/model.bin path/to/dataset.tsv method path/to/vectorized')

path_to_model = 'vectors/GoogleNews-vectors-negative300.bin' if sys.argv[1] == 'default' else sys.argv[1]
path_to_dataset = sys.argv[2]
method = sys.argv[3]
path_to_vectorized = sys.argv[3]

# Load desired word model.
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)  

# Read in data to vectorize
df = pd.read_csv(path_to_dataset, sep='\t', header=None)

# Replace cols 1 and 2 with their vectors using the model and gensim
words_0 = df[0]
words_1 = df[1]
y = df[[2]]

def vectorize_word(word):
	if word in model.wv:
		return pd.Series(model.wv[word])
	else:
		# The pretrained word2vec model has dimensionality 300
		return pd.Series([np.nan] * 300)

vectors_0 = words_0.apply(vectorize_word)
vectors_1 = words_1.apply(vectorize_word)

# Concat
vectors_x = pd.concat([vectors_0, vectors_1, y], axis = 1)

vectors_x.to_csv(path_to_vectorized, sep='\t', header=False, index=False)