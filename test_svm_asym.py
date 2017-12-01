from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

clf = joblib.load('models/svm.pkl')
test = pd.read_csv('lexical_entailment/bless2011/data_lex_test.tsv', sep='\t', header=None)
test_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_test_vectorized.tsv', sep='\t', header=None)
val_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_val_vectorized.tsv', sep='\t', header=None)

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

	true_count = 0
	for p in y:
		if p:
			true_count += 1
	print true_count, orig_rows

	num_correct = accuracy_score(y, preds, normalize=False)



	print test_name, ": percentage non-nan correct:", num_correct/float(test_df.shape[0])
	print test_name, ": percentage correct overall", num_correct/float(orig_rows)

evaluate('test')
