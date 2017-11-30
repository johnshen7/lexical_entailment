import gensim 
import pandas as pd
import numpy as np

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('vectors/GoogleNews-vectors-negative300.bin', binary=True)  

# Train
train = pd.read_csv('lexical_entailment/bless2011/data_lex_train.tsv', sep='\t', header=None)

# Test
# test = pd.read_csv('lexical_entailment/bless2011/data_lex_test.tsv', sep='\t', header=None)

# Replace cols 1 and 2 with their vectors using the model and gensim
words_0 = train[0]
words_1 = train[1]
y = train[[2]]

def vectorize_word(word):
	if word in model.wv:
		return pd.Series(model.wv[word])
	else:
		return pd.Series([np.nan] * 300)

vectors_0 = words_0.apply(vectorize_word)
vectors_1 = words_1.apply(vectorize_word)

vectors_x = pd.concat([vectors_0, vectors_1, y], axis = 1)
vectors_x.to_csv('lexical_entailment/bless2011/data_lex_test_vectorized.tsv', sep='\t', header=False, index=False)