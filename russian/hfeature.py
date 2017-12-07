from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.externals import joblib
import sklearn.metrics as metrics
import pandas as pd
import numpy as np

# Open vectorized file
df = pd.read_csv('lrwc_vectorized.tsv', sep='\t', header=None)

train, test = train_test_split(df.values)
print train.shape, test.shape
train = pd.DataFrame(train)
test = pd.DataFrame(test)

### Training
train.dropna(axis=0, inplace=True)
X = train.iloc[:, :600]
y = train.iloc[:, -1].astype(bool)

H = train.iloc[:, :300]
w = train.iloc[:, 300:600]

iterations = 2

feature_vector = pd.DataFrame()

for _ in range(iterations):
    # Fit log reg
    clf = LogReg()
    clf.fit(X, y)
    print "Log reg fit"

    # Decision plane -- "H-feature detector"
    p = np.array([clf.coef_[0][300:600]])
    print "found p"

    # Generate feature vector
    # cos(H, w)
    hw_sim = pd.DataFrame(np.dot(H.values, w.values.T).diagonal())
    print "hw done"
    # cos(H, p)
    hp_sim = pd.DataFrame(np.dot(H.values, p.T))
    print "hp done"
    # cos(w, p)
    wp_sim = pd.DataFrame(np.dot(w.values, p.T))
    print "wp done"
    # cos((H - w), p)
    hwp_sim = pd.DataFrame(np.dot((H.values - w.values), p.T))
    print "hwp done"

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
print "Training complete"

### Testing
orig_rows, orig_cols = test.shape

# Remove rows with NaN
test.dropna(axis=0, inplace=True)

# Count number of rows removed
diff = orig_rows - test.shape[0]

X = test.iloc[:, :-1]
y = test.iloc[:, -1].astype(bool)

H = test.iloc[:, :300]
w = test.iloc[:, 300:600]

feature_vector = pd.DataFrame()

for _ in range(iterations):
    # Fit log reg
    h_clf = LogReg()
    h_clf.fit(X, y)
    print "Log reg fit"

    # Decision plane -- "H-feature detector"
    p = np.array([h_clf.coef_[0][300:600]])

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
    w_proj = np.dot(np.dot(w, p.T / p.sum()), p)
    w_rej = np.array(w - w_proj)
    w_rej_norm = w_rej / w_rej.sum(axis=1).reshape(w_rej.shape[0], -1)
    w = pd.DataFrame(w_rej_norm)
    X = pd.concat([H, w], axis = 1)

preds = final_clf.predict(feature_vector)

print "precision", metrics.precision_score(y, preds)
print "recall", metrics.recall_score(y, preds)
print "f1", metrics.f1_score(y, preds)

num_correct = metrics.accuracy_score(y, preds, normalize=False)

print "test : percentage non-nan correct:", num_correct/float(test.shape[0])
print "test : percentage correct overall", num_correct/float(orig_rows)
