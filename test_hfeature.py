from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

clf = joblib.load('models/svm_hfeature.pkl') 

test_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_test_vectorized.tsv', sep='\t', header=None)
val_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_val_vectorized.tsv', sep='\t', header=None)

# Test and validation in one go bc i'm lazy
for test_name, test_df in zip(['test', 'val'], [test_vectorized, val_vectorized]):
	orig_rows, orig_cols = test_df.shape

	print test_df.shape
	# Remove rows with NaN
	test_df.dropna(axis=0, inplace=True)

	print test_df.shape

	# Count number of rows removed
	diff = orig_rows - test_df.shape[0]

	X = test_df.iloc[:, :-1]
	y = test_df.iloc[:, -1]

	H = test_df.iloc[:, :300]
	w = test_df.iloc[:, 300:600]

	iterations = 3

	feature_vector = pd.DataFrame()

	for _ in range(iterations):
        # Fit log reg
		h_clf = LogReg()
		h_clf.fit(X, y)

		# Decision plane -- "H-feature detector"
		p = np.array([h_clf.coef_[0][300:600]])

		# Generate feature vector
		# cos(H, w)
		hw_sim = pd.DataFrame(np.dot(H.values, w.values.T).diagonal())
		# cos(H, p)
		hp_sim = pd.DataFrame(np.dot(H.values, p.T))
		# cos(w, p)
		wp_sim = pd.DataFrame(np.dot(w.values, p.T))
		# cos((H - w), p)
		hwp_sim = pd.DataFrame(np.dot((H.values - w.values), p.T))

		# Union with previous feature vectors
		new_df = pd.concat([feature_vector, hw_sim, hp_sim, wp_sim, hwp_sim], axis = 1)
		print "new df dims:", new_df.shape
		feature_vector = new_df

		# Generate new <H, w> pairs
		h_proj = np.dot(np.dot(H, p.T / p.sum()), p)
		h_rej = np.array(H - h_proj)
		h_rej_norm = h_rej / h_rej.sum(axis=1).reshape(h_rej.shape[0], -1)
		H = pd.DataFrame(h_rej_norm)
		w_proj = np.dot(np.dot(w, p.T / p.sum()), p)
		w_rej = np.array(w - w_proj)
		w_rej_norm = w_rej / w_rej.sum(axis=1).reshape(w_rej.shape[0], -1)
		w = pd.DataFrame(w_rej_norm)
		X = pd.concat([H, w], axis = 1)

	preds = clf.predict(feature_vector)

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