from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import pandas as pd

# Open vectorized files
train = pd.read_csv('../datasets/bless2011/data_lex_train_vectorized.tsv', sep='\t', header=None)
test = pd.read_csv('../datasets/bless2011/data_lex_test_vectorized.tsv', sep='\t', header=None)


### Training
# Remove NaN
train.dropna(axis=0, inplace=True)
X = train.iloc[:, :600]
y = train.iloc[:, -1].astype(bool)

clf = LogReg()
clf.fit(X, y)

### Testing
orig_rows, orig_cols = test.shape

# Remove rows with NaN
test.dropna(axis=0, inplace=True)

# Count number of rows removed
diff = orig_rows - test.shape[0]

X = test.iloc[:, :600]
y = test.iloc[:, -1].astype(bool)

preds = clf.predict(X)

print "precision", metrics.precision_score(y, preds)
print "recall", metrics.recall_score(y, preds)
print "f1", metrics.f1_score(y, preds)

num_correct = metrics.accuracy_score(y, preds, normalize=False)

print "test : percentage non-nan correct:", num_correct/float(test.shape[0]) 
print "test : percentage correct overall", num_correct/float(orig_rows)
