import pandas as pd
import numpy as np

words = pd.read_csv('lrwc_no_nan.tsv', sep='\t')

print words.shape
print words.columns

num_w1 = np.unique(words.iloc[:, 0])
num_w2 = np.unique(words.iloc[:, 1])

print len(num_w1), len(num_w2)

some_words = np.random.choice(num_w1, size=100, replace=False)

#some_words = num_w1[:50]
# Get all the words that these touch

print len(some_words)

df_1 = words[words.isin(some_words).any(axis=1)]

df_2 = words[~(words.isin(some_words).any(axis=1))]
print df_1.shape
print df_2.shape


df_1.to_csv('lrwc_train.tsv', sep='\t', index=False)
df_2.to_csv('lrwc_test.tsv', sep='\t', index=False)