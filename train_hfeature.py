from sklearn.linear_model import LogisticRegression as LogReg
from sklearn import svm
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Open vectorized training file
train = pd.read_csv('lexical_entailment/bless2011/data_lex_train_vectorized.tsv', sep='\t', header=None)

train.dropna(axis=0, inplace=True)
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

H = train.iloc[:, :300]
w = train.iloc[:, 300:600]

iterations = 1

feature_vector = pd.DataFrame()

for _ in range(iterations):
    # Fit log reg
    clf = LogReg()
    clf.fit(X, y)

    # Decision plane -- "H-feature detector"
    p = clf.coef_[0][:300]

    # Generate feature vector
    # cos(H, w)
    hw_sim = pd.DataFrame(np.dot(H.values, w.values.T).diagonal())
    print hw_sim.shape
    # cos(H, p)
    hp_sim = pd.DataFrame(np.dot(H.values, p.T))
    print hp_sim.shape
    # cos(w, p)
    wp_sim = pd.DataFrame(np.dot(w.values, p.T))
    print wp_sim.shape
    # cos((H - w), p)
    hwp_sim = pd.DataFrame(np.dot((H.values - w.values), p.T))
    print hwp_sim.shape

    # Union with previous feature vectors
    new_df = pd.concat([feature_vector, hw_sim, hp_sim, wp_sim, hwp_sim], axis = 1)
    print "new df dims:", new_df.shape
    feature_vector = new_df

    # Generate new <H, w> pairs -- NOT DONEEEEE


# Use SVM on feature vectors for final classifier
final_clf = svm.SVC(class_weight='balanced')
final_clf.fit(feature_vector, y)

joblib.dump(final_clf, 'models/svm_hfeature.pkl')
