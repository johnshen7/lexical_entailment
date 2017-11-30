import gensim 
import sklearn 
import pandas as pd

# Load Google's pre-trained Word2Vec model.
# model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  

# Train
train = pd.read_csv('lexical_entailment/bless2011/data_lex_train.tsv', sep='\t', header=None, nrows=5)

# Replace cols 1 and 2 with their vectors