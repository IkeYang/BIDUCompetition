# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:06:36 2022

@author: wcfda
"""
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice



# ============
# Generate datasets. choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
data_dir = r'...' # directory where data is stored

# wind turbine spatial data
filename_loc = 'sdwpf_baidukddcup2022_turb_location.csv'
data_loc = pd.read_csv(data_dir + filename_loc)
data_loc.drop('TurbID', axis = 1, inplace = True)

# wind speed data
filename_full = 'clean_data.csv'
data_full = pd.read_csv(data_dir + filename_full)
turbID_set = set(data_full['TurbID'])
turbID_list = list(turbID_set)
turbID_list.sort()
data_ws = []
for turbineID in turbID_list:
    ws_full = data_full['Wspd'][data_full['TurbID'] == turbineID]
    # data of day 0 to 7, 60 to 67, 120 to 127 are selected for clustering
    # different time intervals are considered to remove seasonal effect
    ws = ws_full[:7*24*6].tolist() + ws_full[60*24*6:67*24*6].tolist() + ws_full[120*24*6:127*24*6].tolist()
    data_ws.append(ws)

data_ws = pd.DataFrame(data_ws)



# ============
# PCA visulization
# ============
# reduce dimension to 2d for visulizaton
# get some ideas of how wind speed data distribute
# and choose appropriate clustering methods based on its distribution
pca = PCA(n_components = 2)
pca.fit(data_ws)
pca_data = {'PC1':pca.components_[0],
            'PC2':pca.components_[1]}

pca_data = pd.DataFrame(pca_data)
sn.FacetGrid(pca_data, size = 6).map(plt.scatter, 'PC1', 'PC2')



# ============
# Set up cluster parameters
# ============
plot_num = 1

default_base = {
    "quantile": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 5,
    "n_clusters": 3,
    "min_samples": 10
}

# do clustering for both spatial and wind speed data
data_params = [
    (
     data_loc, {
         "quantile": 0.3,
         "damping": 0.9,
         "preference": -200,
         "n_neighbors": 5,
         "n_clusters": 3,
         "min_samples": 10
        }
     ),
    (
     data_ws, {
         "quantile": 0.3,
         "damping": 0.9,
         "preference": -200,
         "n_neighbors": 5,
         "n_clusters": 3,
         "min_samples": 10 
        }
     )
]



# ============
# clustering and plotting
# ============
for i, (dataset, algo_params) in enumerate(data_params):
    
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X = np.array(dataset)

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    # remove nan generated during normalization
    X = pd.DataFrame(X).dropna(axis = 1)
    X = np.array(X)
    
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile = params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors = params["n_neighbors"], include_self = False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Create cluster objects
    ms = cluster.MeanShift(bandwidth = bandwidth, bin_seeding = True)
    two_means = cluster.MiniBatchKMeans(n_clusters = params["n_clusters"])
    ward = cluster.AgglomerativeClustering(
        n_clusters = params["n_clusters"], linkage = "ward", connectivity = connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters = params["n_clusters"],
        eigen_solver = "arpack",
        affinity = "nearest_neighbors",
    )
    affinity_propagation = cluster.AffinityPropagation(
        damping = params["damping"], preference = params["preference"], random_state = 0
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage = "average",
        affinity = "cityblock",
        n_clusters = params["n_clusters"],
        connectivity = connectivity,
    )
    birch = cluster.Birch(n_clusters = params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components = params["n_clusters"], covariance_type = "full"
    )

    clustering_algorithms = (
        ("MiniBatch\nKMeans", two_means),
        ("Affinity\nPropagation", affinity_propagation),
        ("MeanShift", ms),
        ("Spectral\nClustering", spectral),
        ("Ward", ward),
        ("Agglomerative\nClustering", average_linkage),
        ("BIRCH", birch),
        ("Gaussian\nMixture", gmm),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message = "Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category = UserWarning,
            )
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)
        print(y_pred)
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i == 0:
            plt.title(name, size = 5)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        # plot labeled wind turbine spatial location data 
        plt.scatter(data_loc['x'], data_loc['y'], s=10, color=colors[y_pred])

        plt.xlim(-10, 6000)
        plt.ylim(0, 13000)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=10,
            horizontalalignment="right",
        )
        plot_num += 1

plt.show()
# conclusion: wind speed distribution does not strongly depend on where turbines are located
# clustering results for spatial location data and wind speed data are significant distint
 
