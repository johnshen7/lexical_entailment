from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import sys
import sklearn.metrics as metrics

# Open vectorized file
# df = pd.read_csv('../datasets/russian/lrwc_vectorized.tsv', sep='\t', header=None)

# train, test = train_test_split(df.values)
# print train.shape, test.shape
# train = pd.DataFrame(train)
# test = pd.DataFrame(test)

# Open vectorized files
train = pd.read_csv('../datasets/russian/lrwc_vectorized_train.tsv', sep='\t', header=None)
test = pd.read_csv('../datasets/russian/lrwc_vectorized_test.tsv', sep='\t', header=None)

print train.shape, test.shape

### Training
# Remove NaN
train.dropna(axis=0, inplace=True)
X = train.iloc[:, :-1]
y = train.iloc[:, -1].astype(bool)

clf = GaussianNB()
clf.fit(X, y)

print "Fit Gaussian NB"

### Testing 
orig_rows, orig_cols = test.shape

# Remove rows with NaN
test.dropna(axis=0, inplace=True)

# Count number of rows removed
diff = orig_rows - test.shape[0]

X = test.iloc[:, :-1]
y = test.iloc[:, -1].astype(bool)

preds = clf.predict(X)

print "precision", metrics.precision_score(y, preds)
print "recall", metrics.recall_score(y, preds)
print "f1", metrics.f1_score(y, preds)

num_correct = metrics.accuracy_score(y, preds, normalize=False)

print "test : percentage non-nan correct:", num_correct/float(test.shape[0]) 
print "test : percentage correct overall", num_correct/float(orig_rows)