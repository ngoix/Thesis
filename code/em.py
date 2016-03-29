import numpy as np
import pdb


def EM(estimator, X, t, lim_inf=None, lim_sup=None, precision=10,
       n_generated=100000):
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

    p = (n_samples - np.arange(n_samples)) / float(n_samples)

    s_unif_repeated = np.array([s_unif]).repeat(n_samples, axis=0)

    Leb = (s_unif_repeated > s_X[s_X_argsort].reshape(-1, 1)).sum(
        axis=1) / n_generated * volume_support
    EM_t = (p.reshape(-1, 1) - t * Leb.reshape(-1, 1)).max(axis=0)
    pdb.set_trace()
    return EM_t

    # EM_t = np.array([1. / n_samples * (n_samples - i) - t * sum(s_unif > s_X[s_X_argsort[i]]) / n_generated * volume_support for i in range(n_samples)]).max(axis=0)
    # EM_t = np.maximum(np.zeros(t.shape[0]), EM_t)




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


def EM_approx(estimator, X, t, lim_inf=None, lim_sup=None, precision=70):
    '''
    Compute an approximation of the mass-volume curve at point t of the scoring
    function corresponding to 'estimator'
    Parameters:
    estimator: fitted estimator (eg damex.predict)
    X: testing data
    t: array of float
    lim_inf: numpy array of shape(d,) (default is None)
        the infimum of the data support along each dimension
        if None, computed wrt the testing data X only
    lim_sup: numpy array of shape(d,) (default is None)
        the supremum of the data support along each dimension
        if None, computed wrt the testing data X only
    '''
    n_samples, n_features = X.shape
    if lim_inf is None:
        lim_inf = X.min(axis=0) - 1e-7
    # 1e-7 since otherwise (lim_sup - lim_inf - precision * step).max() > 0
    # so that some data are not in any cell
    if lim_sup is None:
        lim_sup = X.max(axis=0) + 1e-7
    feature_range = precision
    step = (lim_sup - lim_inf) / precision
    volume_cell = step.prod()
    nb_cell = feature_range ** n_features
    print 'nb_cell=', nb_cell
    approx_scoring = np.zeros(nb_cell)
    n_samples_cell = np.zeros(nb_cell)
    print 'first loop with length nb_cell'

    decision_func_X = estimator.decision_function(X)
    cell_dict = {}
    for i in range(n_samples):
        x = X[i, :]
        y = decision_func_X[i]
        base_ = np.array((x - lim_inf) / step, dtype=int)
        num_cell = nombre(base_, n_features)
        if num_cell in cell_dict:
            cell_dict[num_cell] += np.array([1, y])
        else:
            cell_dict[num_cell] = np.array([1, y])
    nb_non_empty_cell = len(cell_dict)
    approx_scoring = np.array([cell_dict[k][1] / cell_dict[k][0]
                               for k in cell_dict])
    n_samples_cell = np.array([cell_dict[k][0] for k in cell_dict])

    if n_samples_cell.sum() != n_samples:
        raise ValueError("some data are not in any cell"
                         " need to increase 1e-7 in lim_inf and lim_sup")
    print 'sorting the approx:'
    a_s_argsort = approx_scoring.argsort()
    print 'precomputing'
    p = n_samples_cell[a_s_argsort[::-1]].cumsum(dtype=float) / n_samples
    print 'computing EM avoiding for loop'
    EM_t = np.array([p[i] - t * volume_cell * i
                     for i in range(0, nb_non_empty_cell)]).max(axis=0)
    print 'last operation'
    EM_t = np.maximum(np.zeros(t.shape[0]), EM_t)
    return EM_t

    # for i in range(nb_cell):
    #     base_ = base(i, feature_range)
    #     while len(base_) < n_features:
    #         base_ = [0] + base_
    #     inf = [lim_inf[j] + int(base_[j]) * step[j]
    #            for j in range(n_features)]
    #     sup = inf + step
    #     index = (X <= sup).all(axis=1) * (X >= inf).all(axis=1)
    #     n_samples_cell[i] = index.sum()
    #     if index.sum() == 0:
    #         approx_scoring[i] = -np.inf
    #     else:
    #         approx_scoring[i] = estimator.decision_function(X[index]).mean()

    # if n_samples_cell.sum() != n_samples:
    #     raise ValueError("some data are not in any cell"
    #                      " need to increase 1e-7 in lim_inf and lim_sup")
    # print 'sorting the approx:'
    # a_s_argsort = approx_scoring.argsort()
    # # set first just before the first non-empty cell:
    # print 'precomputing'
    # first = np.where(n_samples_cell[a_s_argsort[:]] > 0)[0][0] - 1
    # p = n_samples_cell[a_s_argsort[first:]].cumsum(dtype=float) / n_samples
    # print 'computing EM avoiding for loop'
    # EM_t = np.array([p[i] - t * volume_cell * i for i in range(0, nb_cell - first)]).max(axis=0)
    # print 'last operation'
    # EM_t = np.maximum(np.zeros(t.shape[0]), EM_t)
    # pdb.set_trace()
    # return EM_t

    ########################
    # EM_t = np.zeros(t.shape[0])
    # for i in range(0, nb_cell - first):
    #     EM_t = np.maximum(EM_t, p[i] - t * volume_cell * i)


    # EM_tmp = 0
    # for i in reversed(range(t.shape[0])):
    #     stop = False
    #     while stop is not True:
    #         cpt += 1
    #         p_tmp = p + float(n_samples_cell[a_s_argsort[cpt]]) / n_samples
    #         EM_tmp = p_tmp - t[i] * volume_cell * cpt
    #         if EM_tmp < EM_t[i] or cpt >= nb_cell - 1:
    #             stop = True
    #             cpt -= 1
    #         else:
    #             EM_t[i] = EM_tmp
    #             p = p_tmp


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


def base(n, b):
    if n == 0:
        return []
    else:
        return base(n // b, b) + [n % b]

    
def nombre(a, b):
    n = 0
    cpt = 0
    for j in range(a.shape[0]):
        cpt += 1
        n += a[j] * b ** cpt
    return n
