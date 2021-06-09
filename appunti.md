# APPUNTI
___________________ 
## GAUSSIAN CLASSIFIERS
Assumendo di non normalizzare i dati di Training e di Evaluation, testando Gaussian, NaiveBayes e TiedCovariance Classifier si evidenzia che il NaiveBayes e il TiedCovariance classifier performano peggio rispetto al Gaussian classifier.

## LDA
Con l'LDA la distribuzione delle caratteristiche dei campioni è uguale. Pertanto non c'è alcuna informazione utile da ricavare.

## PCA
Con la PCA su 2 dimensioni si distinguono abbastanza bene i cluster di 0 e 1, ma meglio quelli dello 0. Se la capacità di riconoscere 0 fosse più importante, la PCA sembrerebbe fare un buon lavoro. Si nota che lungo la terza componente principale la separazione è minima.

## NORMALIZZAZIONE-Z
Assumendo di avere dei dati di Training e Evaluation normalizzati secondo la Normalizzazione-Z, si nota che che l'errore dell Gaussian e del NaiveBayes peggiorano, mentre quello del TiedCovariance risulta essere il migliore finora.