import pickle
import numpy as np

from matplotlib import pyplot as plt, cm
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))
print(dataset)

configurations = [['single', 'euclidean'], ['complete', 'euclidean'], ['single', 'cosine'], ['complete', 'cosine']]
for c in configurations:
    cluster_list = AgglomerativeClustering(linkage=c[0], affinity=c[1], compute_distances=True).fit(dataset)
    plot_dendrogram(cluster_list, truncate_mode="level", p=5)
    plt.title("Hierarchical clustering dendrogram with " + c[0] + " " + c[1])
    plt.xlabel("Number of points in node")
    plt.savefig('dendrogram_for_' + c[0] + "_" + c[1])
    plt.show()
    range_n_clusters = [2, 3, 4, 5, 6]
    for n_clusters in range_n_clusters:
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 12)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(dataset) + (n_clusters + 1) * 10])
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=c[0], affinity=c[1],
                                            compute_distances=True).fit(dataset)
        cluster_labels = clusterer.fit_predict(dataset)
        silhouette_avg = silhouette_score(dataset, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        sample_silhouette_values = silhouette_samples(dataset, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.set_title("The silhouette plot for the %d clusters." % n_clusters + c[0] + " " + c[1])
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.savefig('figure%d_' % n_clusters + c[0] + "_" + c[1])
    plt.show()
