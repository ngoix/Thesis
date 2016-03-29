import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc
from sklearn.datasets import fetch_kddcup99
from em import EM, MV, EM_approx

np.random.seed(1)


datasets = ['http']  # , 'http']

for dat in datasets:
    # loading and vectorization
    print('loading data')
    if dat in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True, percent10=True)
        X = dataset.data
        y = dataset.target

    y = (y != 'normal.').astype(int)

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X = X.astype(float)
    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    axis_t = np.arange(0, 10, 0.01)
    axis_alpha = np.arange(0, 1, 0.01)

    # fit:
    print('LocalOutlierFactor processing...')
    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit(X_train)

    print('IsolationForest processing...')
    iforest = IsolationForest()
    iforest.fit(X_train)

    # EM:
    plt.subplot(121)
    # em_lof = EM_approx(lof, X_test, axis_t)
    # AUC = auc(axis_t, em_lof)
    # plt.plot(axis_t, em_lof, lw=1, label='EM-curve of %s for %s (area = %0.3f)'
    #          % ('lof', dat, AUC))
    print 'em_iforest...'
    em_iforest = EM(iforest, X_test, axis_t)
    AUC = auc(axis_t, em_iforest)
    plt.plot(axis_t, em_iforest, lw=1,
             label='EM-curve of %s for %s (area = %0.3f)'
             % ('iforest', dat, AUC))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('t')
    plt.ylabel('EM(t)')
    plt.title('Excess-Mass curve')
    plt.legend(loc="lower right")

    # # MV:
    # plt.subplot(122)
    # mv_lof = MV(lof, X_test, axis_alpha)
    # AUC = auc(axis_alpha, mv_lof)
    # plt.plot(axis_alpha, mv_lof, lw=1,
    #          label='MV-curve of %s for %s (area = %0.3f)'
    #          % ('lof', dat, AUC))

    # mv_iforest = MV(iforest, X_test, axis_alpha)
    # AUC = auc(axis_alpha, mv_iforest)
    # plt.plot(axis_alpha, mv_iforest, lw=1,
    #          label='MV-curve of %s for %s (area = %0.3f)'
    #          % ('iforest', dat, AUC))
    # plt.xlim([-0.05, 1.05])
    # plt.xlabel('alha')
    # plt.ylabel('MV(alpha)')
    # plt.title('Mass-Volume Curve')
    # plt.legend(loc="lower right")

plt.show()
