import math
import scipy
import numpy
import matplotlib.pyplot as plt
from scipy import linalg, special, stats
from numpy import genfromtxt


def mcol(X):
    return X.reshape((len(X), 1))

def mrow(X):
    return X.reshape((1, len(X)))

def loadFile(filename):
    my_data = genfromtxt(filename, delimiter=',')

    return my_data[:, 0:-1].T, numpy.array(my_data[:, -1], dtype='int64')


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def MU_Cov_calculator(Data, label):  # accept matrix of dim (num_feature, num_sample)

    # return a two dictionary mu, Cov, that contain mu[label], Cov[label]
    mu = {}
    C = {}
    for i in numpy.unique(label):
        DataTmp = Data[:, label == i]
        mu[i] = DataTmp.mean(axis=1).reshape(Data.shape[0], 1)
        B = DataTmp - mu[i]
        C[i] = (1 / DataTmp.shape[1]) * numpy.dot(B, B.T)

    return mu, C

def GAU_logpdf_ND(x, mu, C):

    (sign, log_det)=numpy.linalg.slogdet(C)
    C_inv= numpy.linalg.inv(C)
    Dim=C.shape[0]


    y=[]
    for column in x.T:
        c=column.reshape(column.size,1)
        tmp=(-Dim/2)*numpy.log(2*numpy.pi)-0.5*log_det-0.5*numpy.dot((c-mu).T,C_inv).dot((c-mu))
        t=tmp[0,0]
        y.append(t)
    return numpy.array(y)

def compute_FNR(CM):
    return CM[0, 1]/(CM[0, 1]+CM[1, 1])


def compute_FPR(CM):
    return CM[1, 0]/(CM[1, 0]+CM[0, 0])


def z_normalization(DTR):
    local = numpy.array(DTR)
    mean = local.mean(axis=1)
    std = local.std(axis=1)
    local -= mcol(mean)
    local /= mcol(std)
    return local

def PCAplot(DTR):
    mu = mcol(numpy.mean(DTR, axis=1))

    TMP = DTR-mu
    N = DTR.shape[1]
    C = numpy.dot(TMP, TMP.T)/N

    U, s, Vh = numpy.linalg.svd(C)

    normalizedEigenvalues = s / numpy.sum(s)
    #print(per_var)
    labels = ['PC' + str(x) for x in range(1, len(s) + 1)]
    plt.bar(x=range(1, len(s) + 1), height=normalizedEigenvalues, tick_label=labels)
    plt.ylabel('percentage of Variance per PC')
    plt.xlabel('Principal Component')
    plt.title('PC variance plot')
    plt.show()

    return U


def PCA(DTR, LTR, U, m=None):
    
    if(m==None):
        m = 2 
    P = U[:, 0:m]
    y = numpy.dot(P.T, DTR)
    data = {}

    for i in numpy.unique(LTR):
        data[i] = y[:, LTR==i]

    plt.figure()
    plt.xlabel('PC1')
    plt.ylabel('PC2')


    for i in numpy.unique(LTR):
        plt.scatter(data[i][0, :], data[i][1, :], label=i)

    plt.legend()
    plt.show()
    
    return y

def gaussianize(DTR):
    M = numpy.zeros((DTR.shape[0], DTR.shape[1]))

    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            tmp = DTR[i, :] < DTR[i, j]
            M[i, j] = (numpy.sum(tmp)+1)/(DTR.shape[1]+2)

    res = scipy.stats.norm.ppf(M)

    # for i in range(DTR.shape[0]):
    #     plt.figure()
    #     plt.hist(res[i, :], bins=50, ec='black')
    #     plt.show()

    return res


def compute_confusion_matrix(predictedLabels, testLabels):
    taskDim=len(numpy.unique(testLabels))

    CM = numpy.zeros((taskDim, taskDim))
        
    for i in range(len(predictedLabels)):
        CM[predictedLabels[i]][int(testLabels[i])] += 1

    return CM

def compute_optimal_B_decision(app, llr, testlabels, t=None):
    
    pt, Cfn, Cfp= app
    if(t == None):
        t = -numpy.log((pt * Cfn) / ((1 - pt) * Cfp)) #soglia

    predictedL = numpy.zeros(len(llr), dtype=int)
    for i in range(len(llr)):
        if (llr[i] > t):
            predictedL[i] = 1
        else:
            predictedL[i] = 0
    
    CM = compute_confusion_matrix(predictedL, testlabels)

    # print(CM)
    # print('\n\n')

    return CM


def compute_Bayes_risk(CM, app): ##empirical Bayes Risk A recognizer that has lower cost will provide more accurate answers
    p1, Cfn, Cfp = app
    FNR = compute_FNR(CM)
    FPR = compute_FPR(CM)

    return ((p1*Cfn*FNR)+((1-p1)*Cfp*FPR))


def compute_norm_Bayes(unNormBayesRisk, app):
    p1, Cfn, Cfp = app
    BDummy = min(p1*Cfn, (1-p1)*Cfp)

    return unNormBayesRisk/BDummy

def compute_min_DCF(llr, app, LTE):
   
    DCF_list= []
    treshold= numpy.sort(llr)
    
    for i in range(len(LTE)):
        t=treshold[i]
        tmp=compute_Bayes_risk(compute_optimal_B_decision(app, llr, LTE, t), app)
        DCF_list.append(compute_norm_Bayes(tmp, app))
        
    return min(DCF_list), treshold[numpy.argmin(DCF_list)]
  
    
def logreg_obj_wrap(DTR, LTR, l):

    def logreg_obj(v):
        n = DTR.shape[1]
        w, b = v[0:-1], v[-1]
        w = w.reshape((len(w), 1))
        J = 0
        x = DTR
        for idx in range(n):
            if LTR[idx] == 0:
                c = 0
            else:
                c = 1

            J += (c * numpy.log1p(numpy.exp(numpy.dot(-w.T, x[:, idx]) - b)) + (1 - c) * numpy.log1p(numpy.exp(numpy.dot(w.T, x[:, idx]) + b)))

        return l/2 * ((numpy.linalg.norm(w))**2) + 1/n * J

    return logreg_obj

## SVM

def compute_H_hat(D_hat, LTR):

    G = numpy.dot(D_hat.T, D_hat)
    z = numpy.equal.outer(LTR, LTR)
    z = 2*z-1
    G = z*G

    return G


def compute_w(D_hat, LTR, alpha):

    z = LTR*2-1
    D = z*D_hat
    w = numpy.dot(alpha, D.T)

    return w

def J_primal(w, C, DTR, LTR):

    s = 0
    z = LTR*2-1
    for i in range(DTR.shape[1]):
        s += max(0, 1-z[i]*numpy.dot(w, DTR[:, i]))
    v = 1/2*((linalg.norm(w))**2) + C * s

    return v

def compute_H_hat2(DTR, LTR, psi, c=None, d=None, gamma=None):

    z = numpy.equal.outer(LTR, LTR)
    z = 2*z-1
    if(gamma != None and c == None and d == None):
        H = z * radial_basis_kernel(DTR, gamma, psi)
    elif(c != None and d != None and gamma == None):
        H = z*polynomial_kernel(DTR, d, c, psi)
    else:
        print('\nInvalid parameter\n')

    return H

def radial_basis_kernel(DTR, gamma, psi, DTE=None, computeScore=False, alpha=None, z = None):

    if(computeScore == False):
        nCol = DTR.shape[1]
        M = numpy.zeros((nCol, nCol))

        for i in range(DTR.shape[1]):
            k = numpy.exp(-gamma * (linalg.norm((-DTR +mcol(DTR[:, i])), axis=0) ** 2))+psi
            M[i, :] += k
    else:
        nCol = DTE.shape[1]
        M = numpy.zeros((nCol))

        for i in range(DTR.shape[1]):
            a = alpha[i]
            z_ = z[i]
            if (a != 0):
                k = numpy.exp(-gamma*(linalg.norm((-DTE+mcol(DTR[:, i])), axis=0)**2))+psi
                M += a * z_ * k

    return M

def polynomial_kernel(DTR, d, c, psi=None, DTE=None, computeScore=False, alpha=None, z=None):

    if(computeScore == False):
        return (numpy.dot(DTR.T, DTR) + c)**d + psi
    else:
        nCol = DTE.shape[1]
        M = numpy.zeros((nCol))
        
        k = ((numpy.dot(DTR.T, DTE)+c)**d) + psi
        for i in range(DTR.shape[1]):
            a = alpha[i]
            z_ = z[i]
            if(a != 0):
                M += a*z_*k[i, :]
                
        
        return M
    
def compute_score(alpha, DTR, LTR, DTE, psi, c=None, d=None, gamma=None):
    z = LTR*2-1

    if (gamma != None):
        S = radial_basis_kernel(DTR, gamma, psi, DTE, True, alpha, z)
        #S = numpy.sum(S, axis=0)
    elif (c != None and d != None):
        S = polynomial_kernel(DTR, d, c, psi, DTE, True, alpha, z)
        #S = numpy.sum(S, axis=0)
    else:
        print('\nInvalid parameter\n')

    return S

def EM(X, ref_GMM, resps):
    N = X.shape[1]

    
    ref_GMM = comp_stats(X, resps, ref_GMM)
    S = comp_score_GMM(X, ref_GMM)
    logMargDen = scipy.special.logsumexp(S, axis=0)
    succLL = numpy.sum(logMargDen)/N
    resps = numpy.exp(S-logMargDen)

    return ref_GMM, S, succLL, resps


def comp_stats(X, responsabilities, gmm):
    statistics = {}
    temporary = []
    sumZ = 0

    for i, g in enumerate(gmm.values()):
        Z = numpy.sum(mrow(responsabilities[i, :]), axis=1)
        F, S = comp_stats_F_and_S(responsabilities[i, :], X)
        temporary.append((Z, F, S))
        sumZ += Z
    
    for i in range(len(gmm)):
        Z, F, S = temporary[i]
        newMu = F/Z
        newC = (S/Z)-numpy.dot(newMu, newMu.T)
        newW = Z/sumZ
        statistics[i] = [newW, newMu, newC]

    return statistics.copy()

def comp_stats_F_and_S(respRelatedToComp, X):
    
    S = numpy.zeros((X.shape[0], X.shape[0]), dtype='float64')
    F = numpy.zeros((X.shape[0], 1))
   
    for i in range(X.shape[1]):
        S += respRelatedToComp[i]*numpy.dot(mcol(X[:, i]), mrow(X[:, i]))
        F += respRelatedToComp[i]*mcol(X[:, i])

    return F, S

def comp_score_GMM(X, gmm):
    M = len(gmm)
    N = X.shape[1]

    S = numpy.zeros((M, N), dtype='float64')

    for i, g in enumerate(gmm.values()):
        #print(g)
        prior = g[0]
        mu = g[1]
        C = g[2]
        S[i, :] += GAU_logpdf_ND(X, mu, C)+numpy.log(prior)

    return S

def optimize_GMM(X, GMM, t=1e-6):
    N = X.shape[1]
    
    S = comp_score_GMM(X, GMM)
    logMargDen = scipy.special.logsumexp(S, axis=0)
    prevLL = numpy.sum(logMargDen)/N
    resps = numpy.exp(S-logMargDen)

    GMM, S, succLL, resps = EM(X, GMM, resps)
    
    while(succLL-prevLL >= t):
        prevLL = succLL
        GMM, S, succLL, resps  = EM(X, GMM, resps)
        
    return GMM, S