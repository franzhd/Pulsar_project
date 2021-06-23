import numpy as np
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


def PCA(DTR, LTR, U):

    m = 3
    P = U[:, 0:m]
    y = numpy.dot(P.T, DTR)
    data = {}

    for i in numpy.unique(LTR):
        data[i] = DTR[:, LTR==i]

    plt.figure()
    plt.axes(projection='3d')
    plt.xlabel('PC1')
    plt.ylabel('PC2')


    for i in numpy.unique(LTR):
        plt.scatter(data[i][0, :], data[i][1, :], data[i][2, :], label=i)

    plt.legend()
    plt.show()


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
    trashold= numpy.sort(llr)
    
    for i in range(len(LTE)):
        t=trashold[i]
        tmp=compute_Bayes_risk(compute_optimal_B_decision(app, llr, LTE, t), app)
        DCF_list.append(compute_norm_Bayes(tmp, app))
        
    return min(DCF_list)