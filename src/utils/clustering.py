"""
Clustering utilities module for embedding-based clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def perform_clustering(features, method='dbscan', eps=0.5, n_clusters=None):
    """
    Clusters embeddings using specified method.

    Args:
        features: array-like, shape (n_samples, n_features)
        method: str, 'dbscan' or 'agglomerative'
        eps: float, DBSCAN parameter for maximum distance between samples
        n_clusters: int, number of clusters for AgglomerativeClustering
    
    Returns:
        labels: array, shape (n_samples,) Cluster labels for each sample
    """
    if method == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=4, metric='cosine')
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    
    labels = clusterer.fit_predict(features)
    return labels

def visualize_cluster_accuracy(cluster_id, pred_labels, dataset, max_imgs=8):
    """
    Plots images from a cluster and shows their real ID to check for errors.

    Args:
        cluster_id: int, ID of the cluster to visualize
        pred_labels: array-like, shape (n_samples,) Predicted cluster labels
        dataset: Dataset object, used to retrieve images and true IDs
        max_imgs: int, maximum number of images to display from the cluster
    """
    cluster_indices = np.where(pred_labels == cluster_id)[0]
    
    plt.figure(figsize=(16, 4))
    for i, idx in enumerate(cluster_indices[:max_imgs]):
        img, true_pid, _ = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        
        plt.subplot(1, max_imgs, i+1)
        plt.imshow(img_np)
        plt.title(f"True ID: {true_pid}")
        plt.axis('off')
        
    plt.suptitle(f"Analysis of Cluster {cluster_id} (Predicted as one person)", fontsize=16)
    plt.show()

def calculate_cluster_purity(true_labels, pred_labels):
    """
    Calculates the percentage of the majority class in each cluster.

    Args:
        true_labels: array-like, shape (n_samples,) Ground truth cluster labels
        pred_labels: array-like, shape (n_samples,) Predicted cluster labels
    Returns:
        float: Average purity score across all clusters
    """
    purity_scores = []
    unique_clusters = np.unique(pred_labels[pred_labels != -1])
    
    for c in unique_clusters:
        pids_in_cluster = true_labels[pred_labels == c]
        majority_count = np.max(np.bincount(pids_in_cluster))
        purity = majority_count / len(pids_in_cluster)
        purity_scores.append(purity)
    
    return np.mean(purity_scores)

def evaluate_clusters(true_labels, pred_labels):
    """
    Computes NMI and ARI. Ground truth is used only for validation.

    Args:
        true_labels: array-like, shape (n_samples,) Ground truth cluster labels
        pred_labels: array-like, shape (n_samples,) Predicted cluster labels
    
    Returns:
        dict: Dictionary containing NMI and ARI scores
    """
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return {"NMI": nmi, "ARI": ari}