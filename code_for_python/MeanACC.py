import numpy as np

def compute_similarity(W, var):
    if W.shape[0] < W.shape[1]:
        W = W.T
    n = W.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            y1 = W[i, :]
            y2 = W[j, :]
            sij = np.exp(-np.square(np.linalg.norm(y1-y2, ord=2)/var))
            S[i, j] = sij
    return S


def meanacc(W, labels):
    S0 = compute_similarity(W, 0.5)
    S = S0 - np.diag(np.diag(S0))
    if labels.shape[0] < labels.shape[1]:
        labels = labels.T

    U = np.unique(labels)
    indexX = list()
    lenX = np.zeros((U.shape[0], 1))
    for i in range(U.shape[0]):
        indexI = np.where(labels == U[i])
        lenX[i, 0] = indexI[0].shape[0]
        lenX = lenX.astype(int)
        indexX.append(indexI[0])

    SortedIdx = np.argsort(-S, axis=1)
    for i in range(labels.shape[0]):
        labelI = labels[i, 0]
        locaI = np.where(U == labelI)
        locaI = locaI[0]

        inSet = np.intersect1d(indexX[locaI[0]], SortedIdx[i, 0:lenX[locaI[0], 0]])
        retriACC = inSet.shape[0]/lenX[locaI]
    meanAcc = np.mean(retriACC)
    return meanAcc
