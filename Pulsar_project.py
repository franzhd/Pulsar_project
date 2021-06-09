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


def LDA(DTR, LTR):
    n_features = DTR.shape[0]
    SW = numpy.zeros((n_features, n_features))
    SB = numpy.zeros((n_features, n_features))

    media_globale = mcol(numpy.mean(DTR, axis=1))

    for i in numpy.unique(LTR):
        sample = DTR[:, LTR == i]
        media_local = mcol(numpy.mean(sample, axis=1))
        n_sample = sample.shape[1]
        SW += (sample - media_local).dot((sample - media_local).T)
        media_diff = (media_local - media_globale)
        SB += n_sample * (media_diff).dot((media_diff).T)

    SW = SW / DTR.shape[1]
    SB = SW / DTR.shape[1]

    s, U = scipy.linalg.eigh(SB, SW)

    per_var = s / numpy.sum(s)
    lab = ['PC' + str(x) for x in range(1, len(s) + 1)]
    plt.bar(x=range(1, len(s) + 1), height=per_var, tick_label=lab)
    plt.ylabel('percentage of Variance per PC')
    plt.xlabel('Principal Component')
    plt.title('LDA variance plot')
    plt.show()


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


class GaussianClassifier:
    def __init__(self):
        self.C = {}
        self.mu = {}

    def train(self, DTR, LTR):
        self.mu, self.C = MU_Cov_calculator(DTR, LTR)

    def test(self, DTE, LTE):
        S = numpy.zeros((numpy.unique(LTE).size, DTE.shape[1]))
        predicted = []

        for i in numpy.unique(LTE):
            S[i, :] = numpy.exp(GAU_logpdf_ND(DTE, self.mu[i], self.C[i]) + numpy.log(1 / 2))

        Sp = scipy.special.logsumexp(S, axis=0)

        for x, p in zip(S.T, Sp):
            # print(x, x.shape)
            tmp = x - p
            predicted.append(numpy.argmax(tmp))

        predicted = numpy.array(predicted)

        True_prediction = numpy.array([predicted == LTE])

        error = 1 - (numpy.count_nonzero(True_prediction) / True_prediction.size)

        print("Gaussian Classifier error:", error)


class TiedCovClassifier:
    def __init__(self):
        self.mu = {}

    def train(self, DTR, LTR):
        self.mu, C = MU_Cov_calculator(DTR, LTR)
        N = DTR.shape[1]
        self.C_ = numpy.zeros((DTR.shape[0], DTR.shape[0]))
        for i in numpy.unique(LTR):
            Nc = DTR[:, LTR == i].shape[1]
            self.C_ += Nc*C[i]
        self.C_ /= N


    def test(self, DTE, LTE):
        S = numpy.zeros((numpy.unique(LTE).size, DTE.shape[1]))
        predicted = []

        for i in numpy.unique(LTE):
            S[i, :] = numpy.exp(GAU_logpdf_ND(DTE, self.mu[i], self.C_) + numpy.log(1 / 2))

        Sp = scipy.special.logsumexp(S, axis=0)

        for x, p in zip(S.T, Sp):
            # print(x, x.shape)
            tmp = x - p
            predicted.append(numpy.argmax(tmp))

        predicted = numpy.array(predicted)

        True_prediction = numpy.array([predicted == LTE])

        error = 1 - (numpy.count_nonzero(True_prediction) / True_prediction.size)

        print("TiedCovClassifier error:", error)


class BayesClassifier:
    def __init__(self):
        self.C = {}
        self.mu = {}
        
    def train(self, DTR, LTR):

        self.mu, self.C = MU_Cov_calculator(DTR, LTR)
        for i in numpy.unique(LTR):
            self.C[i] *= numpy.eye(self.C[i].shape[0])

    def test(self, DTE, LTE):

        S = numpy.zeros((numpy.unique(LTE).size, DTE.shape[1]))
        predicted = []

        for i in numpy.unique(LTE):
            S[i, :] = numpy.exp(GAU_logpdf_ND(DTE, self.mu[i], self.C[i]) + numpy.log(1 / 2))

        Sp = scipy.special.logsumexp(S, axis=0)

        for x, p in zip(S.T, Sp):
            tmp = x - p
            predicted.append(numpy.argmax(tmp))

        predicted = numpy.array(predicted)

        True_prediction = numpy.array([predicted == LTE])

        error = 1 - (numpy.count_nonzero(True_prediction) / True_prediction.size)

        print("Naive Bayes Classifier error:", error)


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

    return None

if __name__ == '__main__':
    fname = 'Train.txt'
    Gauss = GaussianClassifier()
    TiedGauss = TiedCovClassifier()
    Bayes = BayesClassifier()

    Gauss_norm = GaussianClassifier()
    tied_norm = TiedCovClassifier()
    Bayes_norm = BayesClassifier()

    D, L = loadFile(fname)

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    LDA(DTR, LTR)
    eigenvectors = PCAplot(DTR)


    PCA(DTR, LTR, eigenvectors)

    Gauss.train(DTR, LTR)
    Gauss.test(DTE, LTE)

    Gauss_norm.train(z_normalization(DTR), LTR)
    Gauss_norm.test(DTE, LTE)
    Gauss_norm.test(z_normalization(DTE), LTE)

    TiedGauss.train(DTR, LTR)
    TiedGauss.test(DTE, LTE)

    tied_norm.train(z_normalization(DTR), LTR)
    tied_norm.test(DTE, LTE)
    tied_norm.test(z_normalization(DTE), LTE)

    Bayes.train(DTR, LTR)
    Bayes.test(DTE, LTE)

    Bayes_norm.train(z_normalization(DTR), LTR)
    Bayes_norm.test(DTE, LTE)
    Bayes_norm.test(z_normalization(DTE), LTE)

    print('***END***')
