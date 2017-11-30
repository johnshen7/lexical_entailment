from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

clf = joblib.load('models/svm.pkl') 

test_vectorized = pd.read_csv('lexical_entailment/bless2011/data_lex_test_vectorized.tsv', sep='\t', header=None)
orig_rows, orig_cols = test_vectorized.shape

# Remove rows with NaN
test_vectorized.dropna(axis=0, inplace=True)

# Count number of rows removed
diff = orig_rows - test_vectorized.shape[0]

X = test_vectorized.iloc[:, :-1]
y = test_vectorized.iloc[:, -1]

preds = clf.predict(X)

num_correct = accuracy_score(y, preds, normalize=False)

print "percentage non-nan correct:", num_correct/float(test_vectorized.shape[0]) 
print "percentage correct overall", num_correct/float(orig_rows)