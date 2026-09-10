"""
Train a LINEAR readout (logistic regression, no kernel, no hidden layer) on
the downstream flow patches and see if it predicts XOR/OR/AND.

Baselines, so a good number means something:
  1. Linear readout on the RAW INPUT BITS (a,b) -- should fail on XOR by
     construction (not linearly separable), succeed on OR/AND. This is the
     sanity check that the task is actually hard for a plain linear model.
  2. Linear readout on the flow's response, held-out test split, for XOR,
     OR, and AND. If XOR accuracy on held-out trials clears the raw-bit
     baseline (~50%) by a wide margin, the medium's nonlinearity is doing
     real work, not just adding noise.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

d = np.load('xor_data.npz')
X, bits = d['X'], d['bits']
tasks = {'XOR': d['y_xor'], 'OR': d['y_or'], 'AND': d['y_and']}

print("=== Baseline: linear readout on the RAW BITS themselves ===")
bits_f = bits.astype(float)
for name, y in tasks.items():
    clf = LogisticRegression()
    scores = cross_val_score(clf, bits_f, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}  (raw bits, expect ~0.5 for XOR)")

print("\n=== Linear readout on the FLOW's downstream patch (held-out CV) ===")
Xs = StandardScaler().fit_transform(X)
for name, y in tasks.items():
    clf = LogisticRegression(max_iter=2000, C=0.3)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xs, y, cv=cv)
    print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}")

print("\n=== Same, but only 2 features (mean vorticity, mean |vorticity|) ===")
X_small = np.stack([X.mean(axis=1), np.abs(X).mean(axis=1)], axis=1)
X_small = StandardScaler().fit_transform(X_small)
for name, y in tasks.items():
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X_small, y, cv=cv)
    print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}  (2-feature probe, not full patch)")