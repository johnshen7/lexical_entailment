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

iterations = 3

feature_vector = pd.DataFrame()

for _ in range(iterations):
    # Fit log reg
    clf = LogReg()
    clf.fit(X, y)

    # Decision plane -- "H-feature detector"
    p = np.array([clf.coef_[0][:300]])

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
    print H.shape
    w_proj = np.dot(np.dot(w, p.T / p.sum()), p)
    w_rej = np.array(w - w_proj)
    w_rej_norm = w_rej / w_rej.sum(axis=1).reshape(w_rej.shape[0], -1)
    w = pd.DataFrame(w_rej_norm)
    print w.shape
    X = pd.concat([H, w], axis = 1)

print X.shape
# Use SVM on feature vectors for final classifier
final_clf = svm.SVC(class_weight='auto')
final_clf.fit(feature_vector, y)

joblib.dump(final_clf, 'models/svm_hfeature.pkl')
