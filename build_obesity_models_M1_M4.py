import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

DATA_DIR = Path(r'C:/Users/TDG16-220831006323/Downloads')
train = pd.read_csv(DATA_DIR/'train.csv')
test  = pd.read_csv(DATA_DIR/'test.csv')
sample = pd.read_csv(DATA_DIR/'sample_submission.csv')

target = 'NObeyesdad' if 'NObeyesdad' in train.columns else train.columns[-1]
y = train[target].astype(str)
X = train.drop(columns=[target])
X_te = test.copy()

# unify columns (get_dummies makes all features nonnegative counts/0-1)
full = pd.concat([X, X_te], axis=0, ignore_index=True)
cat_cols = [c for c in full.columns if full[c].dtype=='object']
num_cols = [c for c in full.columns if c not in cat_cols]
full_enc = pd.get_dummies(full, columns=cat_cols, dummy_na=False)
X_enc = full_enc.iloc[:len(X)].copy()
X_te_enc = full_enc.iloc[len(X):].copy()

# scale real numeric columns only
scale_cols = [c for c in num_cols if c in X_enc.columns]
if scale_cols:
    sc = StandardScaler()
    X_enc[scale_cols] = sc.fit_transform(X_enc[scale_cols])
    X_te_enc[scale_cols] = sc.transform(X_te_enc[scale_cols])

# detect id column if present in both test and sample
id_col = next((c for c in ['id','ID','Id'] if c in test.columns and c in sample.columns), None)
tgt_cols = [c for c in sample.columns if c.lower()!='id']
sub_target = tgt_cols[0] if tgt_cols else sample.columns[-1]

models = [
    ('Model 1 — Multinomial Logistic Regression', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)),
    ('Model 2 — Linear Discriminant Analysis (shrinkage)', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ('Model 3 — Naïve Bayes (Gaussian)', GaussianNB()),
    ('Model 4 — Linear SVM (One-vs-Rest)', LinearSVC()),
]

def quick_cv(clf, X, y, k=3, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    acc, f1 = [], []
    for tr, va in skf.split(X, y):
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict(X.iloc[va])
        acc.append(accuracy_score(y.iloc[va], p))
        f1.append(f1_score(y.iloc[va], p, average='macro'))
    return float(np.mean(acc)), float(np.mean(f1))

rows, outfiles = [], []
for name, clf in models:
    a, f = quick_cv(clf, X_enc, y)
    rows.append({'model': name, 'accuracy_mean': a, 'macro_f1_mean': f})
    clf.fit(X_enc, y)
    preds = clf.predict(X_te_enc).astype(str)  # string labels => nonnegative
    sub = sample.copy()
    if id_col:
        sub = sub.drop(columns=[sub_target]).merge(pd.DataFrame({id_col: test[id_col].values, sub_target: preds}), on=id_col, how='left')
    else:
        sub[sub_target] = preds
    safe = name.replace('—','-').replace(' ','_').replace('(','').replace(')','')
    out = DATA_DIR / f'{safe}.csv'
    sub.to_csv(out, index=False)
    outfiles.append(str(out))

cv = pd.DataFrame(rows).sort_values('macro_f1_mean', ascending=False)
cv_out = DATA_DIR/'CV_Summary_Models_1_to_4.csv'
cv.to_csv(cv_out, index=False)
print('Created files:')
for p in outfiles: print(p)
print(str(cv_out))

