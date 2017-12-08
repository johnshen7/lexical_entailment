#!/usr/bin/env python
import jieba
import gensim
import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

""" USAGE: ./vectorizing.py path/to/model.bin path/to/dataset.tsv path/to/vectorized method
path/to/model.bin: path to the binary gensim model. If 'default' is passed in, uses Google's pre-trained word2vec model
path/to/dataset.tsv: dataset to vectorize. assumes the last column has prediction
method: right now just 'concat'
path/to/vectorized: desired output location

Example: ./vectorizing.py default datasets/bless2011/data_lex_train.tsv \
datasets/bless2011/data_lex_train_vectorized2.tsv \
concat

./vectorizing.py default datasets/bless2011/data_lex_train.tsv datasets/bless2011/data_lex_train_vectorized_asym.tsv asym
./vectorizing.py default datasets/bless2011/data_lex_test.tsv datasets/bless2011/data_lex_test_vectorized_asym.tsv asym
./vectorizing.py default datasets/bless2011/data_lex_val.tsv datasets/bless2011/data_lex_val_vectorized_asym.tsv asym

./vectorizing.py vectors/wiki.zh.vec datasets/baidu2017/dataset.txt datasets/baidu2017/dataset_vectorized.tsv concat

./vectorizing.py default datasets/bless2011/data_lex_train.tsv datasets/bless2011/data_lex_train_vectorized_diff.tsv diff

"""

if len(sys.argv) != 5:
	raise ValueError('Usage: ./vectorizing.py path/to/model.bin path/to/dataset.tsv method path/to/vectorized')


path_to_model = 'vectors/GoogleNews-vectors-negative300.bin' if sys.argv[1] == 'default' else sys.argv[1]
path_to_dataset = sys.argv[2]
path_to_vectorized = sys.argv[3]
method = sys.argv[4]

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

def merge_vectors(v1, v2, method):
	print "method", method
	# Concat
	if method == 'concat':
		return pd.concat([v1, v2, y], axis = 1)
	elif method == 'diff':
		# Roller 2014 says to normalize the difference
		diff = v1 - v2
		return pd.concat([diff, y], axis = 1)
	elif method == 'asym':
		print "asym"
		# diff
		a = v1 - v2
		# squared diff - can't tell if they mean the mag^2 or ea element sq?
		b = pd.DataFrame(np.dot(a,a))
		return pd.concat([a, b, y], axis = 1)
	elif method == 'cosine':
		#this fails???
		print "cosine"
		cos = pd.DataFrame(cosine_similarity(vectors_0.values, vectors_1.values).diagonal())
		return pd.concat([cos, y], axis = 1)

vectors_x = merge_vectors(vectors_0, vectors_1, method)
vectors_x.to_csv(path_to_vectorized, sep='\t', header=False, index=False)
