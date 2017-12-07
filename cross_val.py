import pandas as pd
import numpy as np
import sys
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LogReg

''' Cross validation, given a path vectorized_file to the file that contains the
vectorized and labeled word pairs. Trains on 4/5th of the data at a time.
Output: 1) average F1 score for cross validation on the same data set
        2) F1 score on the distributionally complementary test set.
'''
vectorized_file = 'datasets/bless2011/data_lex_train_vectorized_asym.tsv'
train = pd.read_csv(vectorized_file, sep='\t', header=None)

train.dropna(axis=0, inplace=True)
X = train.iloc[:, :-1]
y = train.iloc[:, -1].astype(bool)

clf = LogReg(class_weight='balanced')
scores = cross_val_score(clf, X, y, scoring='f1', cv=5)
print scores
print 'average:', sum(scores)/len(scores)

vectorized_file = 'lexical_entailment/bless2011/data_lex_test_vectorized_asym.tsv'
test = pd.read_csv(vectorized_file, sep='\t', header=None)
test.dropna(axis=0, inplace=True)
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]
clf.fit(X,y)
pred = clf.predict(X_test)
print 'test score:', metrics.f1_score(pred, y_test)
