import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()

X = scale(digits.data)
y = digits.target

# number of classifications
k = 10
samples, features = X.shape

def bench_k_means(estim, name, data):
    estim.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % 
        (
            name, 
            estim.inertia_,
            metrics.homogeneity_score(y, estim.labels_),
            metrics.completeness_score(y, estim.labels_),
            metrics.v_measure_score(y, estim.labels_),
            metrics.adjusted_rand_score(y, estim.labels_),
            metrics.adjusted_mutual_info_score(y, estim.labels_),
            metrics.silhouette_score(data, estim.labels_, metric='euclidean')
        )
    )

clf = KMeans(n_clusters=k, init='random', n_init=10)
bench_k_means(clf, '1', X)