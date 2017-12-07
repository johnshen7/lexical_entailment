from sklearn import svm
from sklearn.externals import joblib
import pandas as pd
import sklearn.metrics as metrics
# Open vectorized training file
train = pd.read_csv('datasets/bless2011/data_lex_train_vectorized_asym.tsv', sep='\t', header=None)

# Remove NaN
train.dropna(axis=0, inplace=True)
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

clf = svm.SVC(class_weight='auto')
clf.fit(X, y)

joblib.dump(clf, 'models/svm_asym.pkl')


test_vectorized = pd.read_csv('datasets/bless2011/data_lex_test_vectorized_asym.tsv', sep='\t', header=None)
val_vectorized = pd.read_csv('datasets/bless2011/data_lex_val_vectorized_asym.tsv', sep='\t', header=None)

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


	print "precision", metrics.precision_score(y, preds)
	print "recall", metrics.recall_score(y, preds)
	print "f1", metrics.f1_score(y, preds)
	print "True", metrics.accuracy_score(y[y == 1], preds[y == 1])
	print "False", metrics.accuracy_score(y[y == 0], preds[y == 0])
	print true_count, preds_count, orig_rows

	num_correct = metrics.accuracy_score(y, preds, normalize=False)

	print test_name, ": percentage non-nan correct:", num_correct/float(test_df.shape[0])
	print test_name, ": percentage correct overall", num_correct/float(orig_rows)

#evaluate('test')
evaluate('test')
evaluate('val')

