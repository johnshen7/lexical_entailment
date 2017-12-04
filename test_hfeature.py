from sklearn.externals import joblib
import pandas as pd
import sklearn.metrics as metrics

clf = joblib.load('models/svm_hfeature.pkl') 

test_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_test_vectorized.tsv', sep='\t', header=None)
val_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_val_vectorized.tsv', sep='\t', header=None)

# Test and validation in one go bc i'm lazy
for test_name, test_df in zip(['test', 'val'], [test_vectorized, val_vectorized]):
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
			dif.append(test_df.iloc[i])
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