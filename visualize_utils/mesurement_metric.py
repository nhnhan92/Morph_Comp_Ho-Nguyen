from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

rng = np.random.RandomState(1)

score_funcs = [
    ("V-measure", metrics.v_measure_score),
    ("Rand index", metrics.rand_score),
    # ("ARI", metrics.adjusted_rand_score),
    # ("MI", metrics.mutual_info_score),
    ("NMI", metrics.normalized_mutual_info_score),
    # ("AMI", metrics.adjusted_mutual_info_score),
]

def mesurement_metric(X, y, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    kmeans.fit_predict(X)
    results = {}
    for _name, score_func in score_funcs:
        results[_name] = score_func(kmeans.labels_, y)
        
    return results

if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
                [10, 2], [10, 4], [10, 0]])
    y =  np.array([1,1,1,0,1,0])

    print(mesurement_metric(X, y, n_clusters=2))