import numpy as np
import pdb
# import matplotlib.pyplot as plt
# for the cluster to save the fig:
import sys
sys.path.insert(1, '/home/nicolas/Bureau/OCRF')


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc
from sklearn.utils import shuffle as sh
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.datasets import fetch_spambase, fetch_annthyroid, fetch_arrhythmia
from sklearn.datasets import fetch_pendigits, fetch_pima, fetch_wilt
from sklearn.datasets import fetch_internet_ads, fetch_adult
from em import EM, MV  # , EM_approx, MV_approx, MV_approx_over
from sklearn.preprocessing import LabelBinarizer


np.random.seed(1)

# TODO: find good default parameters for every datasets
# TODO: make an average of ROC curves over 10 experiments
# TODO: idem in bench_lof, bench_isolation_forest (to be launch from master)
#       bench_ocsvm (to be created), bench_ocrf (to be created)

# # datasets available:
# datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt',  # 'internet_ads',
#             'adult']

# # continuous datasets:
# datasets = ['http', 'smtp', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'adult']
# new: ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia', 'pendigits',
#       'pima', 'wilt', 'adult']

datasets = ['http', 'smtp', 'shuttle', # 'spambase',
            'pendigits', 'pima', 'wilt', 'adult']
# datasets = ['shuttle']
plt.figure(figsize=(25, 13))

for dat in datasets:
    print 'dataset:', dat
    # loading and vectorization
    print('loading data')

    if dat == 'adult':
        dataset = fetch_adult(shuffle=True)
        X = dataset.data
        y = dataset.target
        # anormal data are those with label >50K:
        y = np.all((y != ' <=50K', y != ' <=50K.'), axis=0).astype(int)

    if dat == 'internet_ads':  # not adapted to oneclassrf
        dataset = fetch_internet_ads(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 'ad.').astype(int)

    if dat == 'wilt':
        dataset = fetch_wilt(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 'w').astype(int)

    if dat == 'pima':
        dataset = fetch_pima(shuffle=True)
        X = dataset.data
        y = dataset.target

    if dat == 'pendigits':
        dataset = fetch_pendigits(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 4).astype(int)
        # anomalies = class 4

    if dat == 'arrhythmia':
        dataset = fetch_arrhythmia(shuffle=True)
        X = dataset.data
        y = dataset.target
        # rm 5 features containing some '?' (XXX to be mentionned in paper)
        X = np.delete(X, [10, 11, 12, 13, 14], axis=1)
        y = (y != 1).astype(int)
        # normal data are then those of class 1

    if dat == 'annthyroid':
        dataset = fetch_annthyroid(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y != 3).astype(int)
        # normal data are then those of class 3

    if dat == 'spambase':
        dataset = fetch_spambase(shuffle=True)
        X = dataset.data
        y = dataset.target

    if dat == 'ionosphere':
        dataset = fetch_mldata('ionosphere')
        X = dataset.data
        y = dataset.target
        X, y = sh(X, y)
        y = (y != 1).astype(int)

    if dat in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True, percent10=False)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        X, y = sh(X, y)
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print('vectorizing data')

    if dat == 'SF':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != 'normal.').astype(int)

    if dat == 'SA':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        lb.fit(X[:, 2])
        x2 = lb.transform(X[:, 2])
        lb.fit(X[:, 3])
        x3 = lb.transform(X[:, 3])
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != 'normal.').astype(int)

    if dat == 'http' or dat == 'smtp':
        y = (y != 'normal.').astype(int)

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X = X.astype(float)
    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    volume_support = (X_test.max(axis=0) - X_test.min(axis=0)).prod()

    axis_t = np.arange(0, 100. / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(0.9, 0.99, 0.001)

    # fit:
    print('IsolationForest processing...')
    iforest = IsolationForest()
    iforest.fit(X_train)
    print('LocalOutlierFactor processing...')
    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit(X_train)
    print('OneClassSVM processing...')
    ocsvm = OneClassSVM()
    ocsvm.fit(X_train)

    # EM:
    print 'em_iforest'
    em_iforest = EM(iforest, X_test, axis_t)
    amax_iforest = np.argmax(em_iforest <= 0.9)
    pdb.set_trace()
    AUC_iforest = auc(axis_t[:amax_iforest], em_iforest[:amax_iforest])
    print 'em_lof'
    em_lof = EM(lof, X_test, axis_t)
    amax_lof = np.argmax(em_lof <= 0.9)
    AUC_lof = auc(axis_t[:amax_lof], em_lof[:amax_lof])
    print 'em_ocsvm'
    em_ocsvm = EM(ocsvm, X_test, axis_t)
    amax_ocsvm = np.argmax(em_ocsvm <= 0.9)
    AUC_ocsvm = auc(axis_t[:amax_ocsvm], em_ocsvm[:amax_ocsvm])

    plt.subplot(121)
    amax = max(amax_iforest, amax_lof, amax_ocsvm)
    plt.plot(axis_t[:amax], em_iforest[:amax], lw=1,
             label='%s for %s dataset (em_score = %0.3e)'
             % ('iforest', dat, AUC_iforest))
    plt.plot(axis_t[:amax], em_lof[:amax], lw=1,
             label='%s for %s (area = %0.3e)'
             % ('lof', dat, AUC_lof))
    plt.plot(axis_t[:amax], em_ocsvm[:amax], lw=1,
             label='%s for %s (area = %0.3e)'
             % ('ocsvm', dat, AUC_ocsvm))

    # plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('t')
    plt.ylabel('EM(t)')
    plt.title('Excess-Mass curve')
    plt.legend(loc="lower right")

    # MV:
    plt.subplot(122)
    print 'mv_iforest'
    mv_iforest = MV(iforest, X_test, axis_alpha)
    AUC = auc(axis_alpha, mv_iforest)
    plt.plot(axis_alpha, mv_iforest, lw=1,
             label='%s (area = %0.3e)'
             % ('iforest', dat, AUC))
    print 'mv_lof'
    mv_lof = MV(lof, X_test, axis_alpha)
    AUC = auc(axis_alpha, mv_lof)
    plt.plot(axis_alpha, mv_lof, lw=1,
             label='%s (area = %0.3e)'
             % ('lof', dat, AUC))
    print 'mv_ocsvm'
    mv_ocsvm = MV(ocsvm, X_test, axis_alpha)
    AUC = auc(axis_alpha, mv_ocsvm)
    plt.plot(axis_alpha, mv_ocsvm, lw=1,
             label='%s (area = %0.3e)'
             % ('ocsvm', dat, AUC))

    plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 100])
    plt.xlabel('alpha')
    plt.ylabel('MV(alpha)')
    plt.title('Mass-Volume Curve for' + dat)
    plt.legend(loc="upper left")

    plt.savefig('mv_em' + dat)
