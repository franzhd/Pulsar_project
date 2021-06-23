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
            S[i, :] = GAU_logpdf_ND(DTE, self.mu[i], self.C[i]) + numpy.log(1 / 2)

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
            S[i, :] = GAU_logpdf_ND(DTE, self.mu[i], self.C_) + numpy.log(1 / 2)

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
        ll=numpy.zeros((numpy.unique(LTE).size, DTE.shape[1]))
        predicted = []

        for i in numpy.unique(LTE):
            ll[i, :]=GAU_logpdf_ND(DTE, self.mu[i], self.C[i])
            S[i, :] =ll[i, :]  + numpy.log(1/2)

        Sp = scipy.special.logsumexp(S, axis=0)

        for x, p in zip(S.T, Sp):
            tmp = x - p
            predicted.append(numpy.argmax(tmp))

        predicted = numpy.array(predicted)
        app = (0.5, 1, 1)
        app_bayes_risk=compute_Bayes_risk(compute_confusion_matrix(predicted, LTE, 2), app)

        DCF = compute_norm_Bayes(app_bayes_risk, app)
        llr = numpy.array(ll[1, :]-ll[0, :])
        minDCF= compute_min_DCF(llr, app, LTE)

        #opt_bayes_risk = compute_Bayes_risk(compute_optimal_B_decision(app, llr_, LTE), app)

       # minDCF = compute_norm_Bayes(opt_bayes_risk, app)

        True_prediction = numpy.array([predicted == LTE])
        error = 1 - (numpy.count_nonzero(True_prediction) / True_prediction.size)
        print("\-/ \-/ \-/ \-/ \-/ ")
        print("Naive Bayes Classifier error:", error)
        print(app,"DCF:", DCF, "minDCF:", minDCF)
        print("/-\ /-\ /-\ /-\ /-\ ")
        


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


def compute_confusion_matrix(predictedLabels, testLabels, taskDim):
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

    CM = compute_confusion_matrix(predictedL, testlabels, 2)

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
        
    for i in range(DTE.shape[1]):
        tmp=compute_Bayes_risk(compute_optimal_B_decision(app, llr, LTE, trashold[i]), app)
        DCF_list.append(compute_norm_Bayes(tmp, app))
        
    return min(DCF_list)
    
    

if __name__ == '__main__':
    fname = 'Train.txt'
    Gauss = GaussianClassifier()
    Gauss2 = GaussianClassifier()
    TiedGauss = TiedCovClassifier()
    TiedGauss2 = TiedCovClassifier()
    Bayes = BayesClassifier()
    Bayes2 = BayesClassifier()

    Gauss_norm = GaussianClassifier()
    tied_norm = TiedCovClassifier()
    Bayes_norm = BayesClassifier()

    D, L = loadFile(fname)

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    #LDA(DTR, LTR)
    #eigenvectors = PCAplot(DTR)

    #PCA(DTR, LTR, eigenvectors)
    DTR_gaussianized = gaussianize(DTR)
    DTE_gaussianized = gaussianize(DTE)


    Gauss.train(DTR, LTR)
    Gauss.test(DTE, LTE)

    Gauss_norm.train(z_normalization(DTR), LTR)
    Gauss_norm.test(DTE, LTE)
    Gauss_norm.test(z_normalization(DTE), LTE)

    Gauss2.train(DTR_gaussianized, LTR)
    Gauss2.test(DTE_gaussianized, LTE)

    TiedGauss.train(DTR, LTR)
    TiedGauss.test(DTE, LTE)

    tied_norm.train(z_normalization(DTR), LTR)
    tied_norm.test(DTE, LTE)
    tied_norm.test(z_normalization(DTE), LTE)

    TiedGauss2.train(DTR_gaussianized, LTR)
    TiedGauss2.test(DTE_gaussianized, LTE)

    Bayes.train(DTR, LTR)
    Bayes.test(DTE, LTE)

    Bayes_norm.train(z_normalization(DTR), LTR)
    Bayes_norm.test(DTE, LTE)
    Bayes_norm.test(z_normalization(DTE), LTE)

    Bayes2.train(DTR_gaussianized, LTR)
    Bayes2.test(DTE_gaussianized, LTE)

    print('***END***')
