import numpy as np
from scipy.optimize import minimize
from network_enhancement import *

def softmax(input, dim, n1, n2):
    x0 = input.reshape(dim, n2)
    x, w = x0[:, 0:n1], x0[:, n1:n2]
    f_exp = np.exp(np.dot(w.T, x))
    exp_sum = np.sum(f_exp, axis=1, keepdims=True)
    f_sfm = f_exp/exp_sum
    return f_sfm

def kldiv(P, Q):
    eps = np.finfo(np.float64).eps
    kl = np.zeros((P.shape[0], 1))
    for i in range(P.shape[0]):
        a, b = P[i, :], Q[i, :]
        c = a/b
        kl[i] = np.sum(P[i, :] * (np.log(P[i, :]/(Q[i, :]+eps)+eps)))
        # kl[i] = np.sum(P[i, :] * (np.log(P[i, :] / Q[i, :])))
    kld = np.sum(kl)
    return kld

def regular(input, L, coef, dim, n1, n2):
    x0 = input.reshape(dim, n2)
    x = x0[:, 0:n1]
    fun = coef * np.trace(np.matmul(np.matmul(x, L), x.T))
    return fun


def objective(x, S, L, reg, dim, n1, n2):
    f = kldiv(S, softmax(x, dim, n1, n2)) + regular(x, L, reg, dim, n1, n2)
    return f

def obj_jac(x, S, L, reg, dim, n1, n2):
    x0 = x.reshape(dim, n2)
    x1 = x0[:, 0:n1]
    x2 = x0[:, n1:n2]
    S_tilde = softmax(x, dim, n1, n2)
    S_delta = S_tilde - S

    fun_x1 = np.matmul(x2, S_delta) + 2 * reg * np.matmul(x1, L)
    fun_x2 = np.matmul(x1, S_delta.T)

    f = np.concatenate([fun_x1, fun_x2], axis=1)
    f = f.reshape(1, dim*n2)
    return f


def enmugr(dataset, opt, ndim):
    order = 2
    # k = int(np.minimum(20, np.ceil(dataset[0, 0].shape[0]/10)))
    k = opt.k
    nnode = dataset[0, 0].shape[0]
    nview = dataset.shape[0]
    nall = nnode * nview

    con_net = []
    L_con = 0
    for i in range(nview):
        netI = network_enhancement(dataset[i, 0], order, k, opt.alpha)
        con_net.append(netI)

        D = np.diag(netI.sum(axis=1))
        L = D - netI
        L_con += L
    net = np.concatenate([con_net[i] for i in range(nview)], axis=0)/nview

  
    x0 = np.random.uniform(0, 1, size=(1, ndim*(nnode + nall)))


    res = minimize(objective, x0, args=(net, L_con, opt.reg, ndim, nnode, nnode+nall), jac=obj_jac, method='L-BFGS-B',
                       options={'maxiter': 500})
    print(res.x)

    all_feature = res.x.reshape(ndim, nnode+nall)
    common_feature = all_feature[:, 0:nnode]
    return common_feature