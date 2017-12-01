from sklearn.externals import joblib
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sys

try:
	if sys.argv[1] == 'cosine':
		test_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_test_vectorized_cosine.tsv', sep='\t', header=None)
		val_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_val_vectorized_cosine.tsv', sep='\t', header=None)
		clf = joblib.load('models/svm_cosine.pkl')
except:
	test_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_test_vectorized_asym.tsv', sep='\t', header=None)
	val_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_val_vectorized_asym.tsv', sep='\t', header=None)
	clf = joblib.load('models/svm_asym.pkl')


test = pd.read_csv('lexical_entailment/bless2011/data_lex_test.tsv', sep='\t', header=None)


def evaluate(test_name):
	if test_name == 'val':
		test_df = val_vectorized
	else:
		test_df = test_vectorized
	orig_rows, orig_cols = test_df.shape

	# Remove rows with NaN
	test_df.dropna(axis=0, inplace=True)

	# Count number of rows removed
	diff = orig_rows - test_df.shape[0]

	X = test_df.iloc[:, :-1]
	y = test_df.iloc[:, -1]

	preds = clf.predict(X)

	true_count = y.sum()
	preds_count = preds.sum()

	dif = []
	for i in range(len(preds)):
		if preds[i] != y.values[i]:
			print(test.iloc[i])
			dif.append(test.iloc[i])
	print 'number of differences: ', len(dif)

	print "precision", metrics.precision_score(y, preds)
	print "recall", metrics.recall_score(y, preds)
	print "f1", metrics.f1_score(y, preds)
	print "True", metrics.accuracy_score(y[y == 1], preds[y == 1])
	print "False", metrics.accuracy_score(y[y == 0], preds[y == 0])
	print true_count, preds_count, orig_rows

	num_correct = metrics.accuracy_score(y, preds, normalize=False)

	print test_name, ": percentage non-nan correct:", num_correct/float(test_df.shape[0])
	print test_name, ": percentage correct overall", num_correct/float(orig_rows)

evaluate('test')
#evaluate('val')
