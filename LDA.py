
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