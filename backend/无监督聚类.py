import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

def calculate_silhouette_score(data, labels):
    score = silhouette_score(data, labels)
    return score
#轮廓系数 取值范围在[-1, 1]之间，值越接近1表示聚类效果越好。

filename = "Iris.csv"
start_feature = 1
end_feature = 5
data = pd.read_csv(filename)
feature = data.iloc[:,start_feature:end_feature].values


def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

def affinity_propagation_clustering(data):
    affinity_propagation = AffinityPropagation()
    affinity_propagation.fit(data)
    labels = affinity_propagation.labels_
    return labels

def agglomerative_clustering(data, n_clusters=3):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(data)
    return labels

def mean_shift_clustering(data, bandwidth):
    mean_shift = MeanShift(bandwidth=bandwidth)
    labels = mean_shift.fit_predict(data)
    return labels

def bisecting_kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters)
    labels = kmeans.fit_predict(data)
    return labels

def dbscan_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

def optics_clustering(data, min_samples, xi):
    optics = OPTICS(min_samples=min_samples, xi=xi)
    optics.fit(data)
    labels = optics.labels_
    return labels

def birch_clustering(data, n_clusters=3):
    birch = Birch(n_clusters=n_clusters)
    labels = birch.fit_predict(data)
    return labels

def spectral_clustering(data, n_clusters=3):
    spectral = SpectralClustering(n_clusters=n_clusters)
    labels = spectral.fit_predict(data)
    return labels

def gaussian_mixture_clustering(data, n_clusters=3):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels


def affinity_clustering(data, n_clusters=3):
    similarity_matrix = pairwise_distances(data, metric='euclidean')
    affinity = -similarity_matrix
    affinity_propagation = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    labels = affinity_propagation.fit_predict(affinity)
    return labels

def fuzzy_c_means_clustering(data, n_clusters=3, m=2, max_iter=100, error_threshold=1e-4):
    data = np.array(data)
    n_samples = len(data)
    n_features = len(data[0])
    U = np.random.rand(n_samples, n_clusters)
    U /= np.sum(U, axis=1)[:, np.newaxis]
    for _ in range(max_iter):
        U_prev = U.copy()
        centroids = np.dot(U.T, data) / np.sum(U, axis=0)[:, np.newaxis]
        distances = pairwise_distances(data, centroids)
        U = 1 / (distances ** (2 / (m - 1)))
        U /= np.sum(U, axis=1)[:, np.newaxis]
        error = np.linalg.norm(U - U_prev)
        if error < error_threshold:
            break
    cluster_labels = np.argmax(U, axis=1)
    return cluster_labels

label_predict = kmeans_clustering(feature)
print("1.K-Means聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = affinity_propagation_clustering(feature)
print("2.亲和传播聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = agglomerative_clustering(feature)
print("3.凝聚层次聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = mean_shift_clustering(feature,1)
print("4.Mean Shift Clustering聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = bisecting_kmeans_clustering(feature)
print("5.Bisecting K-Means聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = dbscan_clustering(feature,eps=0.4, min_samples=15)
print("6.DBSCAN聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = optics_clustering(feature,15,0.05)
print("7.OPTICS算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = birch_clustering(feature)
print("8.BIRCH聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = spectral_clustering(feature)
print("9.谱聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = gaussian_mixture_clustering(feature)
print("10.高斯混合模型算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = affinity_clustering(feature)
print("11.亲和聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

label_predict = fuzzy_c_means_clustering(feature)
print("12.模糊C均值聚类算法下的轮廓系数：",calculate_silhouette_score(feature,label_predict))

