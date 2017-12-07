import pandas as pd
import numpy as np
import sys
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

train = pd.read_csv('lexical_entailment/bless2011/data_lex_train_vectorized_asym.tsv', sep='\t', header=None)

train.dropna(axis=0, inplace=True)
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

clf = svm.SVC(class_weight='balanced')
scores = cross_val_score(clf, X, y, scoring='f1', cv=5)
print scores
print 'average:', scores/len(scores)
