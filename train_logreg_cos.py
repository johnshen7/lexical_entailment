from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.externals import joblib
import pandas as pd
# Open vectorized training file
train = pd.read_csv('lexical_entailment/bless2011/data_lex_train_vectorized_cosine.tsv', sep='\t', header=None)

# Remove NaN
train.dropna(axis=0, inplace=True)
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

clf = LogReg()
clf.fit(X, y)

joblib.dump(clf, 'models/logreg_cosine.pkl')
