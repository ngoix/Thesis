import numpy as np
import pdb
from sklearn.utils import shuffle
from sklearn.metrics import auc

n_generated = 500000


def EM(estimator, X, s_X, max_features=None, averaging=1,
       n_generated=n_generated):
    '''
    Compute the mass-volume curve at point t of the scoring function
    corresponding to 'estimator'
    Parameters:
    estimator: fitted estimator (eg damex.predict)
    X: testing data
    s_X: estimator.decision_function(X) (the lower, the more abnormal)
    lim_inf: numpy array of shape(d,) (default is None)
        the infimum of the data support along each dimension
        if None, computed wrt the testing data X only
    lim_sup: numpy array of shape(d,) (default is None)
        the supremum of the data support along each dimension
        if None, computed wrt the testing data X only
    max_features: sub-sampling features size (default: no subsampling)
    averaging: the number of experiences on different subsamplings
    '''
    n_samples, n_features = X.shape
    AUC = 0
    for nb_exp in range(averaging):
        if max_features is not None:
            features = shuffle(np.arange(n_features))[:max_features]
            X = X[:, features]
        if max_features is None:
            max_features = n_features

        lim_inf = X.min(axis=0)
        lim_sup = X.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod()

        unif = np.random.uniform(
            lim_inf, lim_sup, size=(n_generated, max_features))
        s_unif = estimator.decision_function(unif)

        t = np.arange(0, 100 / volume_support, 0.01 / volume_support)

        EM_t = em(t, n_samples, volume_support, s_unif, s_X)

        amax = np.argmax(EM_t <= 0.9)
        if amax == 0:
            print('ACHTUNG: 0.9 not achieved. Trying with greater axis_t')
            t = np.arange(0, 10000 / volume_support, 1 / volume_support)
            EM_t = em(t, n_samples, volume_support, s_unif, s_X)
            amax = np.argmax(EM_t <= 0.9)
        if amax == 0:
            print '\n failed to achieve 0.9 \n'
        AUC += auc(t[:amax], EM_t[:amax])
    AUC /= nb_exp
    # return the last EM_t:
    return EM_t, AUC


def em(t, n_samples, volume_support, s_unif, s_X):
    EM_t = np.zeros(t.shape[0])
    min_s = min(s_unif.min(), s_X.min())
    max_s = max(s_unif.max(), s_X.max())
    for u in np.arange(min_s, max_s, (max_s - min_s) / 100):
        if (s_unif >= u).sum() > n_generated / 1000:
            EM_t = np.maximum(EM_t, 1. / n_samples * (s_X >= u).sum() -
                              t * (s_unif >= u).sum() / n_generated
                              * volume_support)
    return EM_t


def MV(estimator, X, s_X, max_features=None, averaging=1,
       n_generated=n_generated):
    '''
    Compute the mass-volume curve at point t of the scoring function
    corresponding to 'estimator'
    Parameters:
    estimator: fitted estimator (eg damex.predict)
    X: testing data
    t: float
    lim_inf: numpy array of shape(d,) (default is None)
        the infimum of the data support along each dimension
        if None, computed wrt the testing data X only
    lim_sup: numpy array of shape(d,) (default is None)
        the supremum of the data support along each dimension
        if None, computed wrt the testing data X only
    '''
    n_samples, n_features = X.shape
    axis_alpha = np.arange(0.9, 0.99, 0.001)
    AUC = 0
    for nb_exp in range(averaging):
        if max_features is not None:
            features = shuffle(np.arange(n_features))[:max_features]
            X = X[:, features]
        if max_features is None:
            max_features = n_features

        lim_inf = X.min(axis=0)
        lim_sup = X.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod()

        unif = np.random.uniform(lim_inf, lim_sup,
                                 size=(n_generated, X.shape[1]))
        s_unif = estimator.decision_function(unif)

        # OneClassSVM decision_func returns shape (n,1) instead of (n,):
        if len(s_unif.shape) > 1:
            s_unif = s_unif.reshape(1, -1)[0]
            s_X = s_X.reshape(1, -1)[0]
        MV = mv(axis_alpha, n_samples, volume_support, s_unif, s_X)
        AUC += auc(axis_alpha, MV)

    # return the last EM_t:
    return MV, AUC


def mv(axis_alpha, n_samples, volume_support, s_unif, s_X):
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        # pdb.set_trace()
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return mv
