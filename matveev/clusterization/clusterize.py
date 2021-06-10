# -*- coding: 1251 -*-


import argparse

from collections import defaultdict
import scipy
import sys

from itertools import  product
from scipy.special import gamma
from scipy.spatial.distance import euclidean
from math import sqrt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KDTree

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma

from itertools import cycle, islice
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, calinski_harabasz_score
from jqm_cvi.jqmcvi import base
from S_Dbw import S_Dbw
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import ranksums, kstest, ks_2samp


def volume(r, m):
    return np.pi ** (m / 2) * r ** m / gamma(m / 2 + 1)


def significant(cluster, h, p):
    max_diff = max(abs(p[i] - p[j]) for i, j in product(cluster, cluster))

    # print(max_diff)
    return max_diff >= h


class WishartClusterization:
    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.significance_level = significance_level  # Significance level

    def fit(self, X):
        kdt = KDTree(X, metric='euclidean')
        print('KDTree has been built')
        #add one because you are your neighb.
        distances, neighbors = kdt.query(X, k = self.wishart_neighbors + 1, return_distance = True)
        neighbors = neighbors[:, 1:]


        distances = distances[:, -1]
        print('distances')
        print(' '.join(list(map(str, distances))))


        indexes = np.argsort(distances)
        print('indexes sorted')
        print(' '.join(list(map(str, indexes))))

        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype = int) - 1

        #index in tuple
        #min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)
        #print('Start clustering')
        counter_to_print = 0
        for index in indexes:

            if counter_to_print % 100000 == 0:
                print(counter_to_print)
                print(' '.join(list(map(str, self.object_labels))), flush=True)

            counter_to_print += 1

            neighbors_clusters =\
                np.concatenate([self.object_labels[neighbors[index]], self.object_labels[neighbors[index]]])
            unique_clusters = np.unique(neighbors_clusters).astype(int)
            unique_clusters = unique_clusters[unique_clusters != -1]


            if len(unique_clusters) == 0:
                self._create_new_cluster(index, distances[index])
            else:
                max_cluster = unique_clusters[-1]
                min_cluster = unique_clusters[0]
                if max_cluster == min_cluster:
                    if self.clusters[max_cluster][-1] < 0.5:
                        self._add_elem_to_exist_cluster(index, distances[index], max_cluster)
                    else:
                        self._add_elem_to_noise(index)
                else:
                    my_clusters = self.clusters[unique_clusters]
                    flags = my_clusters[:, -1]
                    if np.min(flags) > 0.5:
                        self._add_elem_to_noise(index)
                    else:
                        significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                        significan *= self.wishart_neighbors
                        significan /= size
                        significan /= np.power(np.pi, dim / 2)
                        significan *= gamma(dim / 2 + 1)
                        significan_index = significan >= self.significance_level

                        significan_clusters = unique_clusters[significan_index]
                        not_significan_clusters = unique_clusters[~significan_index]
                        significan_clusters_count = len(significan_clusters)
                        if significan_clusters_count > 1 or min_cluster == 0:
                            self._add_elem_to_noise(index)
                            self.clusters[significan_clusters, -1] = 1
                            for not_sig_cluster in not_significan_clusters:
                                if not_sig_cluster == 0:
                                    continue

                                for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                    self._add_elem_to_noise(bad_index)
                                self.clusters_to_objects[not_sig_cluster].clear()
                        else:
                            for cur_cluster in unique_clusters:
                                if cur_cluster == min_cluster:
                                    continue

                                for bad_index in self.clusters_to_objects[cur_cluster]:
                                    self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                self.clusters_to_objects[cur_cluster].clear()

                            self._add_elem_to_exist_cluster(index, distances[index], min_cluster)
        self.labels_ = self.clean_data()
        return self.labels_

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq :  index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype = int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis = 0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)


def volume(r, dim):
    """
    Helper function to calculate volumes of a dim-dimensional spheres with radiuses r
    param r: radiuses of a spheres of shape (n_samples,): ndarray
    param dim: dimensionality of a sphere: int

    Returns:
        volumes: ndarray
    """
    dim_const = (np.pi ** (dim / 2)) / gamma(dim / 2 + 1)
    return dim_const * (r ** dim)


def significant(cluster, h, p):
    """
    Helper function to tell if a cluster is significant
    param cluster: vertices in cluster (indexed): list
    h: height hyperparameter: double
    p: estimated saliency for all points
    """
    p_cluster = p[np.array(cluster)]
    pw_difference = np.abs(p_cluster[:, np.newaxis] - p_cluster)
    max_diff = pw_difference.max()

    # print(max_diff)
    return max_diff >= h


class ClusterValidation:
    def __init__(self, data, labels, dist='euclidean'):
        """
        @param data: ndarray of shape (n_samples, n_features)
        @param labels: ndarray of shape (n_samples,), cluster labels for each sample
        @param dist: distance metric to use: str or callable; defaults to 'euclidean'
        """
        self.data = data
        self.labels = labels
        self.dist = dist
        self.num_cluster = labels.max() + 1
        self.cluster_list = list(range(self.num_cluster))

    def RMSSTD(self):
        dof = self.data.shape[0] - self.num_cluster
        n_features = self.data.shape[1]
        result = 0
        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            center = cluster_data.mean(axis=0)
            result += ((cluster_data - center) ** 2).sum()

        return (result / (dof * n_features)) ** (1 / 2)

    def RS(self):
        all_mean = ((self.data - self.data.mean(axis=0)) ** 2).sum()
        part_mean = 0

        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            center = cluster_data.mean(axis=0)
            part_mean += ((cluster_data - center) ** 2).sum()

        return (all_mean - part_mean) / all_mean

    def hubert(self):
        dist_matrix = squareform(pdist(self.data, self.dist))
        n = self.data.shape[0]

        centers = np.empty((self.num_cluster, self.data.shape[1]))
        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            center = cluster_data.mean(axis=0)
            centers[cluster] = center
        center_dist = squareform(pdist(centers, self.dist))

        result = 0
        for i in range(n):
            for j in range(n):
                result += (dist_matrix[i, j] * center_dist[self.labels[i], self.labels[j]])
        return (result * 2) / (n * (n-1)) 

    def CH(self):
        return calinski_harabasz_score(self.data, self.labels)

    def I_index(self):
        all_mean = np.sqrt(((self.data - self.data.mean(axis=0)) ** 2).sum())
        part_mean = 0
        centers = np.empty((self.num_cluster, self.data.shape[1]))

        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            center = cluster_data.mean(axis=0)
            centers[cluster] = center
            part_mean += np.sqrt(((cluster_data - center) ** 2).sum())
        center_dist = squareform(pdist(centers, self.dist))

        max_center_dist = center_dist.max()

        return (all_mean * max_center_dist) / (self.num_cluster * part_mean)

    def dunn(self):
        return base.dunn_fast(self.data, self.labels)

    def silhouette(self):
        return silhouette_score(self.data, self.labels)

    def davies_bouldin(self):
        return davies_bouldin_score(self.data, self.labels)

    def xie_beni(self):
        n = self.data.shape[0]
        part_mean = 0

        centers = np.empty((self.num_cluster, self.data.shape[1]))
        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            center = cluster_data.mean(axis=0)
            centers[cluster] = center
            part_mean += ((cluster_data - center) ** 2).sum()
        center_dist = squareform(pdist(centers, self.dist))

        center_dist = center_dist + np.diag(np.full_like(center_dist[0], center_dist.max()))

        return part_mean / (n * (center_dist.min()**2))

    def SD(self):
        scat, dis = 0, 0

        all_var_vector = np.var(self.data, 0)
        all_var = np.sqrt(np.sum(all_var_vector ** 2))
        part_var = 0

        centers = np.empty((self.num_cluster, self.data.shape[1]))
        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            center = cluster_data.mean(axis=0)
            centers[cluster] = center
            part_var_vector = np.var(cluster_data, 0)
            part_var += np.sqrt((part_var_vector ** 2).sum())
        center_dist = squareform(pdist(centers, self.dist))

        scat = part_var / (all_var * self.num_cluster)

        max_dist = center_dist.max()
        min_dist = (center_dist + np.diag(np.full_like(center_dist[0], max_dist))).min()
        sum_dist = np.sum(1.0 / np.sum(center_dist, axis=1))

        dis = (max_dist * sum_dist) / min_dist

        return scat + dis

    def S_Dbw(self):
        centers = np.empty((self.num_cluster, self.data.shape[1]))
        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            center = cluster_data.mean(axis=0)
            centers[cluster] = center

        sdbw = S_Dbw(self.data, self.labels, centers)
        return sdbw.S_Dbw_result()

    def CVNN(self, n_neighbors=10):
        com = 0
        sep = []
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(self.data)
        knbrs = neigh.kneighbors(return_distance=False)

        for cluster in self.cluster_list:
            cluster_data = self.data[self.labels == cluster, :]
            cluster_ind = np.arange(self.data.shape[0])[self.labels == cluster]
            n_i = cluster_data.shape[0]
            if (n_i > 1):
                com += np.sum(pdist(cluster_data, metric=self.dist)) * 2 / (n_i * (n_i-1))
            sep.append(np.isin(knbrs[cluster_ind], cluster_ind, invert=True).sum() / (n_i * n_neighbors))

        sep = np.max(sep)

        return com + sep

    def run_all(self, n_neighbors=10, calc_hubert=True):
        results = {}
        results['rmsstd'] = self.RMSSTD()
        results['r-squared'] = self.RS()
        if calc_hubert:
            results['hubert'] = self.hubert()
        results['ch_index'] = self.CH()
        results['i_index'] = self.I_index()
        try:
            results['dunn'] = self.dunn()
        except ValueError as e:
            print('dunn failed with')
            print(e)
            results['dunn'] = np.nan
        results['silhouette'] = self.silhouette()
        results['davies_bouldin'] = self.davies_bouldin()
        results['xie_beni'] = self.xie_beni()
        results['sd'] = self.SD()
        try:
            results['s_dbw'] = self.S_Dbw()
        except:
            results['s_dbw'] = np.nan
        results['cvnn'] = self.CVNN(n_neighbors)

        return results


def cluster_mean_intracluster(X, y):
    mean_intracluster_dist = []
    for cluster_id in range(len(np.unique(y))):
        X_cluster = X[y == cluster_id]
        if len(X_cluster) < 2:
            mean_intracluster_dist.append(0)
            continue
        cluster_dist = squareform(pdist(X_cluster))
        n = X_cluster.shape[0]
        mean_intracluster_dist.append(2 * cluster_dist.sum() / (n * (n - 1)))
    return mean_intracluster_dist


def cluster_max_intracluster(X, y):
    max_intracluster_dist = []
    for cluster_id in range(len(np.unique(y))):
        X_cluster = X[y == cluster_id]
        if len(X_cluster) < 2:
            continue
        cluster_dist = squareform(pdist(X_cluster))
        max_intracluster_dist.append(cluster_dist.max())
    return max_intracluster_dist


def cluster_centroid_dist(X, y):
    centroid_dist = []
    for cluster_id in range(len(np.unique(y))):
        X_cluster = X[y == cluster_id]
        if len(X_cluster) < 2:
            continue
        centroid = np.mean(X_cluster, axis=0)
        c_dist = cdist(X_cluster, centroid.reshape(1, -1))
        centroid_dist.append(2 * c_dist.mean())
    return centroid_dist


def cluster_rmsstd(X, y):
    rmsstd_centroid_dist = []
    dof = X.shape[0] - len(np.unique(y))
    n_feat = X.shape[1]
    for cluster_id in range(len(np.unique(y))):
        X_cluster = X[y == cluster_id]
        if len(X_cluster) < 2:
            continue
        centroid = np.mean(X_cluster, axis=0)
        centroid_dist = cdist(X_cluster, centroid.reshape(1, -1))
        rmsstd_centroid_dist.append((centroid_dist.sum() / (dof * n_feat)) ** (0.5))

    return rmsstd_centroid_dist


def cluster_sd_scat(X, y):
    sd_scat = []
    all_var_vector = np.var(X, 0)
    all_var = np.sqrt(np.sum(all_var_vector ** 2))
    for cluster_id in range(len(np.unique(y))):
        X_cluster = X[y == cluster_id]
        if len(X_cluster) < 2:
            continue
        part_var_vector = np.var(X_cluster, 0)
        sd_scat.append(np.sqrt((part_var_vector ** 2).sum()))
    return sd_scat


def cluster_silhouette(X, y):
    result = []
    n_clusters = len(np.unique(y))
    for cluster_id in range(n_clusters):
        X_cluster = X[y == cluster_id]
        if len(X_cluster) < 2:
            continue
        
        cluster_dist = squareform(pdist(X_cluster))
        n = X_cluster.shape[0]
        a = cluster_dist.sum(axis=1) / (n - 1)
        # other clusters
        b = None

        for other_cluster_id in range(n_clusters):
            if other_cluster_id == cluster_id:
                continue
            X_other_cluster = X[y == other_cluster_id]
            between_cluster_dist = cdist(X_cluster, X_other_cluster)
            mean_between = between_cluster_dist.mean(axis=1)

            if b is None:
                b = mean_between
            else:
                b = np.minimum(b, mean_between)
        
        result.append(np.mean((b - a) / np.maximum(a, b)) / n_clusters)

    return result


def cluster_db(X, y):
    n_clusters = len(np.unique(y))
    intra_dists = np.empty(n_clusters)
    centroids = np.empty((n_clusters, len(X[0])), dtype=float)

    for cluster_id in range(n_clusters):
        X_cluster = X[y == cluster_id]
        centroid = np.mean(X_cluster, axis=0)
        centroids[cluster_id] = centroid
        intra_dists[cluster_id] = np.mean(cdist(X_cluster, centroid.reshape(1, -1)))

    centroid_dist = squareform(pdist(centroids))

    centroid_dist[centroid_dist == 0] = np.inf
    intra_dists_between_clusters = intra_dists[:, None] + intra_dists
    scores = np.max(intra_dists_between_clusters / centroid_dist, axis=1)

    return scores


def cluster_cvnn_sep(X, y, n_neighbors=10):
    n_clusters = len(np.unique(y))
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    knbrs = neigh.kneighbors(return_distance=False)

    sep = []

    for cluster_id in range(n_clusters):
        X_cluster = X[y == cluster_id]
        cluster_ind = np.arange(X.shape[0])[y == cluster_id]
        n_i = X_cluster.shape[0]
        
        sep.append(np.isin(knbrs[cluster_ind], cluster_ind, invert=True).sum() / (n_i * n_neighbors))

    return sep


def run_all_stat_tests(X_real, y_real, X_gen, y_gen):
    wilcoxon_res = {}
    ks_res = {}
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    # sizes
    _, real_sizes = np.unique(y_real, return_counts=True)
    _, gen_sizes = np.unique(y_gen, return_counts=True)
    wilcoxon_res['sizes'] = ranksums(real_sizes, gen_sizes)
    ks_res['sizes'] = ks_2samp(real_sizes, gen_sizes)

    sns.kdeplot(x=real_sizes, label='real', ax=axs[0][0])
    sns.kdeplot(x=gen_sizes, label='gen', ax=axs[0][0])
    axs[0][0].set_title('Sizes of clusters')
    axs[0][0].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['sizes'].pvalue,
                       ks_res['sizes'].pvalue),
                   transform=axs[0][0].transAxes)
    axs[0][0].legend()

    # mean intracluster
    stat_real = cluster_mean_intracluster(X_real, y_real)
    stat_gen = cluster_mean_intracluster(X_gen, y_gen)
    wilcoxon_res['mean_dist'] = ranksums(stat_real, stat_gen)
    ks_res['mean_dist'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[0][1])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[0][1])
    axs[0][1].set_title('Mean intracluster distance')
    axs[0][1].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['mean_dist'].pvalue,
                       ks_res['mean_dist'].pvalue),
                   transform=axs[0][1].transAxes)
    axs[0][1].legend()

    # max intracluster
    stat_real = cluster_max_intracluster(X_real, y_real)
    stat_gen = cluster_max_intracluster(X_gen, y_gen)
    wilcoxon_res['max_dist'] = ranksums(stat_real, stat_gen)
    ks_res['max_dist'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[0][2])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[0][2])
    axs[0][2].set_title('Max intracluster distance')
    axs[0][2].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['max_dist'].pvalue,
                       ks_res['max_dist'].pvalue),
                   transform=axs[0][2].transAxes)
    axs[0][2].legend()

    # centroid dist
    stat_real = cluster_centroid_dist(X_real, y_real)
    stat_gen = cluster_centroid_dist(X_gen, y_gen)
    wilcoxon_res['centroid_dist'] = ranksums(stat_real, stat_gen)
    ks_res['centroid_dist'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[1][0])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[1][0])
    axs[1][0].set_title('Centroid distance')
    axs[1][0].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['centroid_dist'].pvalue,
                       ks_res['centroid_dist'].pvalue),
                   transform=axs[1][0].transAxes)
    axs[1][0].legend()

    # rmsstd compactness
    stat_real = cluster_rmsstd(X_real, y_real)
    stat_gen = cluster_rmsstd(X_gen, y_gen)
    wilcoxon_res['rmsstd'] = ranksums(stat_real, stat_gen)
    ks_res['rmsstd'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[1][1])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[1][1])
    axs[1][1].set_title('RMSSTD Compactness')
    axs[1][1].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['rmsstd'].pvalue,
                       ks_res['rmsstd'].pvalue),
                   transform=axs[1][1].transAxes)
    axs[1][1].legend()

    # sd scattering
    stat_real = cluster_sd_scat(X_real, y_real)
    stat_gen = cluster_sd_scat(X_gen, y_gen)
    wilcoxon_res['sd_scat'] = ranksums(stat_real, stat_gen)
    ks_res['sd_scat'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[1][2])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[1][2])
    axs[1][2].set_title('SD scattering')
    axs[1][2].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['sd_scat'].pvalue,
                       ks_res['sd_scat'].pvalue),
                   transform=axs[1][2].transAxes)
    axs[1][2].legend()

    # silhouette index
    stat_real = cluster_silhouette(X_real, y_real)
    stat_gen = cluster_silhouette(X_gen, y_gen)
    wilcoxon_res['silhouette'] = ranksums(stat_real, stat_gen)
    ks_res['silhouette'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[2][0])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[2][0])
    axs[2][0].set_title('Silhouette index')
    axs[2][0].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['silhouette'].pvalue,
                       ks_res['silhouette'].pvalue),
                   transform=axs[2][0].transAxes)
    axs[2][0].legend()

    # db cluster scores
    stat_real = cluster_db(X_real, y_real)
    stat_gen = cluster_db(X_gen, y_gen)
    wilcoxon_res['db_score'] = ranksums(stat_real, stat_gen)
    ks_res['db_score'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[2][1])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[2][1])
    axs[2][1].set_title('DB cluster scores')
    axs[2][1].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['db_score'].pvalue,
                       ks_res['db_score'].pvalue),
                   transform=axs[2][1].transAxes)
    axs[2][1].legend()

    # cvnn separation
    stat_real = cluster_cvnn_sep(X_real, y_real)
    stat_gen = cluster_cvnn_sep(X_gen, y_gen)
    wilcoxon_res['cvnn_sep'] = ranksums(stat_real, stat_gen)
    ks_res['cvnn_sep'] = ks_2samp(stat_real, stat_gen)

    sns.kdeplot(x=stat_real, label='real', ax=axs[2][2])
    sns.kdeplot(x=stat_gen, label='gen', ax=axs[2][2])
    axs[2][2].set_title('CVNN separation')
    axs[2][2].text(0.01, 0.9,
                   'Wilcoxon pvalue = {:.4f}\nKS pvalue = {:.4f}'.format(
                       wilcoxon_res['cvnn_sep'].pvalue,
                       ks_res['cvnn_sep'].pvalue),
                   transform=axs[2][2].transAxes)
    axs[2][2].legend()

    # plt.tight_layout()
    return fig, wilcoxon_res, ks_res


def get_ngrams_from_vectors(X, vector_transform, n=3):
    """
    :param X: данные (n_samples, n_features)
    :param vector_transform: функция, которая преобразует вектора
    :param n: количество векторов в n-грамме
    :return: массив уникальных n-грамм
    """
    X = np.asarray(X)
    ngrams = []
    for i in range(len(X) - n + 1):
        # e.g.:
        # ngrams.append(encoder(np.asarray(X[i:i + n])).numpy().flatten())
        # ngrams.append(pca.transform(np.asarray(X[i:i + n])).flatten())
        ngrams.append(vector_transform(X[i:i + n]).flatten())
    return np.vstack(ngrams)


def clusterize(ngram_series: np.ndarray, n: int, h: float) -> np.array:
    """
    Кластеризация Уишартом
    :param ngram_series: векторные представления ngram_series
    :param n: параметр для Уишарта
    :param h: параметр для Уишарта
    :return: лейблы кластеров
    """
    wishart = WishartClusterization(n, h)
    labels = wishart.fit(ngram_series)
    return labels


def train_PCA(train_dataset, target_dim):
    print('training pca...', file=sys.stderr, flush=True)
    pca = PCA(target_dim)
    pca.fit(train_dataset)
    print('done training pca!', file=sys.stderr, flush=True)
    return pca


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-T', '--train_dataset')
    # parser.add_argument('-v', '--val_dataset')
    # parser.add_argument('-t', '--test_dataset')

    parser.add_argument('-r', '--embeddings_real')
    parser.add_argument('-g', '--embeddings_gen')

    parser.add_argument('-n', '--ngram', type=int, default=3)
    parser.add_argument('-w', '--wishartn', type=int, default=2)
    parser.add_argument('-s', '--significance', type=float, default=1.)
    parser.add_argument('-d', '--dim', type=int, default=16)
    args = parser.parse_args()

    print(args, '\n', flush=True)

    print('loading data...', file=sys.stderr, flush=True)
    train_dataset = np.load(args.train_dataset)
    embeddings_real = np.load(args.embeddings_real)
    embeddings_gen = np.load(args.embeddings_gen)

    print('training pca...', file=sys.stderr, flush=True)
    pca = train_PCA(train_dataset, args.dim)

    print('getting ngrams...', file=sys.stderr, flush=True)
    ngrams_real = get_ngrams_from_vectors(embeddings_real, vector_transform=pca.transform, n=args.ngram)
    ngrams_gen = get_ngrams_from_vectors(embeddings_gen, vector_transform=pca.transform, n=args.ngram)

    for wishartn in [2, 3, 4]:
        for sig in [0.01, 1]:
            try:
                print(wishartn, sig)
                print('clustering real...', file=sys.stderr, flush=True)
                labels_real = clusterize(ngrams_real, wishartn, sig)
                print(ClusterValidation(ngrams_real, labels_real).run_all(calc_hubert=False), flush=True)

                print('clustering gen...', file=sys.stderr, flush=True)
                labels_gen = clusterize(ngrams_gen, wishartn, sig)
                print(ClusterValidation(ngrams_gen, labels_gen).run_all(calc_hubert=False), flush=True)

                print('processing metrics...', file=sys.stderr, flush=True)
                fig, wilcoxon_res, ks_res = run_all_stat_tests(ngrams_real, labels_real, ngrams_gen, labels_gen)
                fig.savefig('graphs/cmp/' + '_'.join(map(str, [args.wishartn, args.significance, args.ngram, args.dim, 'pca20'])) + '.png')
            except Exception as e:
                print(e)
    
    """
    print('\nreal labels:')
    for label in labels_real:
        print(label)
    print('\ngen labels:')
    for label in labels_gen:
        print(label)
    """
