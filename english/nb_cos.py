from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import sys
import sklearn.metrics as metrics

# Open vectorized file
# Open vectorized files
train = pd.read_csv('../datasets/bless2011/data_lex_train_vectorized.tsv', sep='\t', header=None)
test = pd.read_csv('../datasets/bless2011/data_lex_test_vectorized.tsv', sep='\t', header=None)


### Training
# Remove NaN
train.dropna(axis=0, inplace=True)

vectors_0 = train.loc[:, :299]
vectors_1 = train.loc[:, 300:599]
y = train.iloc[:, -1].reset_index(drop=True).astype(bool)
cos = pd.DataFrame(cosine_similarity(vectors_0.values, vectors_1.values).diagonal()).reset_index(drop=True)

train = pd.concat([cos, y], axis = 1, ignore_index=True)

X = train.iloc[:, :-1]

clf = GaussianNB()
clf.fit(X, y)

print "Fit Gaussian NB"

### Testing
orig_rows, orig_cols = test.shape

# Remove rows with NaN
test.dropna(axis=0, inplace=True)

# Count number of rows removed
diff = orig_rows - test.shape[0]

vectors_0 = test.loc[:, :299]
vectors_1 = test.loc[:, 300:599]
y = test.iloc[:, -1].reset_index(drop=True).astype(bool)
cos = pd.DataFrame(cosine_similarity(vectors_0.values, vectors_1.values).diagonal()).reset_index(drop=True)
test = pd.concat([cos, y], axis = 1, ignore_index=True)
X = test.iloc[:, :-1]

preds = clf.predict(X)
print preds
print "precision", metrics.precision_score(y, preds)
print "recall", metrics.recall_score(y, preds)
print "f1", metrics.f1_score(y, preds)

num_correct = metrics.accuracy_score(y, preds, normalize=False)

print "test : percentage non-nan correct:", num_correct/float(test.shape[0])
print "test : percentage correct overall", num_correct/float(orig_rows)
