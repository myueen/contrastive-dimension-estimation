import numpy as np
import pandas as pd
import skdim
from scipy.linalg import eigh
from sklearn.preprocessing import scale


def id_estimators(df, k):
    # Maximum Likelihood algorithm
    MLE = skdim.id.MLE(K=k).fit(df).dimension_
    # Method Of Moments algorithm
    MOM = skdim.id.MOM().fit(df).dimension_
    L = {
        'MLE': MLE,
        'MOM': MOM,
    }
    return L


def est_V1_V2(X1, X2, d1, d2):
    OUT = {}
    p = X1.shape[1]
    Cx1 = np.cov(X1, rowvar=False)
    Cx2 = np.cov(X2, rowvar=False)
    # eigenvalues python package in increasing order
    val1, vectors1 = eigh(Cx1)
    idx = np.argsort(val1)
    descending_idx = idx[::-1]
    vectors1 = vectors1[:, descending_idx]
    V1 = vectors1[:, 0:d1]
    val2, vectors2 = eigh(Cx2)
    idx_ = np.argsort(val2)
    descending_idx_ = idx_[::-1]
    vectors2 = vectors2[:, descending_idx_]
    V2 = vectors2[:, 0:d2]
    OUT['V1'] = V1
    OUT['V2'] = V2
    return OUT


def sigma1_test_stat(X1, X2, d1, d2):
    OUT = est_V1_V2(X1, X2, d1, d2)
    U = OUT['V1']
    V = OUT['V2']
    M = np.matmul(U.T, V)
    _, cosines, _ = np.linalg.svd(M)
    cosines = np.minimum(1, np.maximum(-1, cosines))
    return cosines[::-1][0]     # first elt of reversed cosines list


def sing_vals(U, V):
    M = np.matmul(U.T, V)
    _, cosines, _ = np.linalg.svd(M)
    cosines = np.minimum(1, np.maximum(-1, cosines))
    return cosines


def boot_test(X1, X2, d1, d2, B):
    X1 = scale(X1, with_mean=True, with_std=False)
    X2 = scale(X2, with_mean=True, with_std=False)
    test_stat = sigma1_test_stat(X1, X2, d1, d2)
    n1 = len(X1)
    n2 = len(X2)
    boot_stats = []
    for j in range(1, B+1):
        print(j)
        idx1 = np.random.choice(range(n1), size=n1, replace=True)
        X1t = X1[idx1, :]
        combined = np.vstack((X1, X2))
        idx2 = np.random.choice(range(n1+n2), size=n2, replace=True)
        X2t = combined[idx2, :]
        boot_stats.append(sigma1_test_stat(X1t, X2t, d1, d2))
    p_value = np.mean(boot_stats < test_stat)
    return {'test_stat': test_stat, 'p_value': p_value}


def CD(X1, X2, d1, d2, epsilon=0.1, B=1000):
    p = X1.shape[1]
    OUT = est_V1_V2(X1, X2, d1, d2)
    singular_vals = sing_vals(OUT['V1'], OUT['V2'])
    singular_vals = singular_vals[::-1]
    L = {}
    L['CD'] = sum(singular_vals < 1 - epsilon) + max(d1 - d2, 0)
    test = boot_test(X1, X2, d1, d2, B)
    L['test_stat'] = test['test_stat']
    L['p_value'] = test['p_value']
    L['singular_vals'] = singular_vals
    L['d1'] = d1
    L['d2'] = d2
    return L


def CDE(fg, bg):
    L1 = id_estimators(fg, 10)
    d1 = round(L1["MOM"])
    L2 = id_estimators(bg, 10)
    d2 = round(L2["MOM"])
    return CD(fg, bg, d1, d2)
