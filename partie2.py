# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:40:27 2020

@author: Amira Benrabah
"""
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm

import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import DistanceMetric

def V():
    df = pd.read_excel(r'C:/Users/Z T R/Desktop/TP2/TRYyy/Algeria-Covid19.xlsx')
    
    # Visualtion des données
    df_goal2 = df[['#City', '#Cases']]
    ind = df_goal2.set_index("#City", inplace = True)


    ## A modified bar graph
    bar = df_goal2.plot(kind='bar',figsize=(30, 16), color = "red", legend = None)
    bar
    plt.yticks(fontsize = 14)
    plt.xticks(ind,fontsize = 8)


    plt.xlabel("#City", fontsize = 25)
    plt.ylabel("#Cases", fontsize = 25)
    plt.title("Cases in Algeria", fontsize=32)

    bar.spines['top'].set_visible(False)
    bar.spines['right'].set_visible(False)
    bar.spines['bottom'].set_linewidth(0.5)
    bar.spines['left'].set_visible(True)
    plt.show()


    df_goal2 = df[['#City', '#Recovered']]
    ind = df_goal2.set_index("#City", inplace = True)


    ## A modified bar graph
    bar = df_goal2.plot(kind='bar',figsize=(30, 16), color = "green", legend = None)
    bar
    plt.yticks(fontsize = 14)
    plt.xticks(ind,fontsize = 8)


    plt.xlabel("#City", fontsize = 25)
    plt.ylabel("#Recovered", fontsize = 25)
    plt.title("Recovered in Algeria", fontsize=32)

    bar.spines['top'].set_visible(False)
    bar.spines['right'].set_visible(False)
    bar.spines['bottom'].set_linewidth(0.5)
    bar.spines['left'].set_visible(True)
    plt.show()


    df_goal2 = df[['#City', '#Death']]
    ind = df_goal2.set_index("#City", inplace = True)


    ## A modified bar graph
    bar = df_goal2.plot(kind='bar',figsize=(30, 16), color = "black", legend = None)
    bar
    plt.yticks(fontsize = 14)
    plt.xticks(ind,fontsize = 8)


    plt.xlabel("#City", fontsize = 25)
    plt.ylabel("#Death", fontsize = 25)
    plt.title("Deaths in Algeria", fontsize=32)

    bar.spines['top'].set_visible(False)
    bar.spines['right'].set_visible(False)
    bar.spines['bottom'].set_linewidth(0.5)
    bar.spines['left'].set_visible(True)
    plt.show()






    df ['#City'] = pd.to_numeric (df ['#City'],errors = 'coerce')
    df = df.replace (np.nan, 0, regex = True)
    o=df.dtypes
    print(o)
    print (df)
    # d=str(df['#City'])
    # d =df['#City'].astype('str')
    
    linked = linkage(df, 'single')
    
    
    labelList = range(1, 50)
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()
    
    dis = DistanceMetric.get_metric('euclidean')
    sh= dis.pairwise(df)
    print(sh)
    
    
    linked = linkage(df, 'complete')
    
    labelList = range(1, 50)
    
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()
    dist = DistanceMetric.get_metric('euclidean')
    so= dist.pairwise(df)
    print(so)
    
    wcss = []
    for i in range(1, 49):
        kmeans = KMeans(n_clusters =i, init="k-means++", max_iter=300, n_init=10)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 49), wcss);
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()
    K = range(1,49)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        meandistortions.append(sum(np.min(cdist(df, kmeans.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Moyenne distortion')
    plt.title('Selecting k with the Elbow Method')
    plt.show()
    
    
    for n_clusters in range (5,15):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(df.iloc[:, 0], df.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    
    plt.show()

def main():
    return V()