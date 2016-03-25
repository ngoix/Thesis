import numpy as np
import pdb


def EM(estimator, X, t, lim_inf=None, lim_sup=None, precision=1000,
       n_generated=10000):
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
    n_samples = X.shape[0]
    if lim_inf is None:
        lim_inf = X.min(axis=0)
    if lim_sup is None:
        lim_sup = X.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    # pdb.set_trace()
    unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, X.shape[1]))
    s_unif = estimator.decision_function(unif)
    s_X = estimator.decision_function(X)  # the lower, the more abnormal
    s_X_argsort = s_X.argsort()
    EM_t = np.zeros(t.shape[0])
    for i in range(n_samples):
        u = s_X[s_X_argsort[i]]
        EM_t = np.maximum(EM_t, 1. / n_samples * (n_samples - i) -
                          t * sum(s_unif > u) / n_generated * volume_support)
    # pdb.set_trace()
    return EM_t



    # n_samples = X.shape[0]
    # if lim_inf is None:
    #     lim_inf = X.min(axis=0)
    # if lim_sup is None:
    #     lim_sup = X.max(axis=0)
    # volume_support = (lim_sup - lim_inf).prod()
    # # pdb.set_trace()
    # unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, X.shape[1]))
    # s_unif = estimator.decision_function(unif)
    # s_X = estimator.decision_function(X)  # the lower, the more abnormal
    # max_s = s_unif.max()
    # min_s = s_X.min()
    # EM_t = np.zeros(t.shape[0])
    # for u in np.arange(min_s, max_s, (max_s - min_s) / precision):
    #     # print 1. / n_samples * sum(s_X > u) - t * sum(s_unif > u) / n_generated * volume_support
    #     EM_t = np.maximum(EM_t, 1. / n_samples * sum(s_X > u) -
    #                       t * sum(s_unif > u) / n_generated * volume_support)
    # # pdb.set_trace()
    # return EM_t


def MV(estimator, X, alpha, lim_inf=None, lim_sup=None, precision=1000,
       n_generated=10000):
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
    n_samples = X.shape[0]
    if lim_inf is None:
        lim_inf = X.min(axis=0)
    if lim_sup is None:
        lim_sup = X.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, X.shape[1]))
    s_unif = np.maximum(estimator.decision_function(unif), -1000)
    s_X = np.maximum(estimator.decision_function(X), -1000)
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    MV = np.zeros(alpha.shape[0])
    for i in range(alpha.shape[0]):
        # pdb.set_trace()
        while mass < alpha[i]:
            print i, mass, alpha[i]
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            # u -= (max_s - min_s) / precision
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        MV[i] = float(sum(s_unif > u)) / n_generated * volume_support
    return MV



    # s_X_argsort = s_X.argsort()
    # max_s = s_unif.max()
    # min_s = s_unif.min()
    # u = min_s
    # mass = 100
    # cpt = -1
    # MV = np.zeros(alpha.shape[0])
    # for i in reversed(range(alpha.shape[0])):
    #     # pdb.set_trace()
    #     while mass > alpha[i]:
    #         print i, mass, alpha[i]
    #         cpt += 1
    #         u = s_X[s_X_argsort[cpt]]
    #         # u -= (max_s - min_s) / precision
    #         mass = 1. / n_samples * (n_samples - cpt - 1)  # sum(s_X > u)
    #     MV[i] = float(sum(s_unif > u)) / n_generated * volume_support
    # return MV
