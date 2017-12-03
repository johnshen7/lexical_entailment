from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.externals import joblib
import pandas as pd
# Open vectorized training file
train1 = pd.read_csv('lexical_entailment/bless2011/data_lex_train_vectorized.tsv', sep='\t', header=None)

train_asym = pd.read_csv('lexical_entailment/bless2011/data_lex_train_vectorized_asym.tsv', sep='\t', header=None)

for train, model_file in zip([train1, train_asym], ['models/logreg.pkl', 'models/logreg_asym.pkl']):
	# Remove NaN
	train.dropna(axis=0, inplace=True)
	X = train.iloc[:, :-1]
	y = train.iloc[:, -1]

	clf = LogReg()
	clf.fit(X, y)

	joblib.dump(clf, model_file)

