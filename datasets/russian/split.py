import pandas as pd
import numpy as np

words = pd.read_csv('lrwc_no_nan.tsv', sep='\t')

print words.shape
print words.columns

num_w1 = np.unique(words.iloc[:, 0])
num_w2 = np.unique(words.iloc[:, 1])

print len(num_w1), len(num_w2)

np.random.seed(20)
some_words = np.random.choice(num_w1, size=100, replace=False)

some_words = num_w1[:50]
# Get all the words that these touch

print len(some_words)

df_1 = words[words.isin(some_words).any(axis=1)]
words_in_df1 = np.concatenate([df_1.iloc[:, 0], df_1.iloc[:,1]])
df_2 = words[~(words.isin(words_in_df1).any(axis=1))]
print df_1.shape
print df_2.shape

df_1_all = np.unique(np.concatenate([df_1.iloc[:,0], df_1.iloc[:,1]]))
df_2_all = np.unique(np.concatenate([df_2.iloc[:,0], df_2.iloc[:,1]]))
print "SETDIFF", np.intersect1d(df_1_all, df_2_all)
assert(len(np.intersect1d(df_1_all, df_2_all)) == 0)

df_1.to_csv('lrwc_train.tsv', sep='\t', index=False)
df_2.to_csv('lrwc_test.tsv', sep='\t', index=False)

df_train_index = df_1.index
df_test_index = df_2.index

for file_name, num_cols in zip(['lrwc_vectorized.tsv', 'lrwc_vectorized_asym.tsv', 'lrwc_vectorized_diff.tsv'], [601, 302, 301]):
	df_vectorized = pd.read_csv(file_name, sep='\t')
	df_vectorized.dropna(inplace=True)

	df_vectorized_train = df_vectorized.ix[df_train_index]
	df_vectorized_test = df_vectorized.ix[df_test_index]


	assert(df_vectorized.shape[1] == num_cols)
	assert(df_vectorized.shape[0] == words.shape[0])
	assert(df_vectorized_train.shape[0] == df_1.shape[0])
	assert(df_vectorized_test.shape[0] == df_2.shape[0])

	split_name = file_name.split('.')
	train_name = split_name[0] + '_train.tsv'
	test_name = split_name[0] + '_test.tsv'

	df_vectorized_train.to_csv(train_name, sep='\t', index=False)
	df_vectorized_test.to_csv(test_name, sep='\t', index=False)

