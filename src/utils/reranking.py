import numpy as np
import torch
from tqdm import tqdm

def compute_jaccard_distance(V_q, V_g):
    """
    Implementation of the Jaccard distance from Zhong et al. (2017).
    Calculates the distance using the min/max formula for weighted vectors.

    Note:
        This is a strict implementation of the Jaccard distance as defined in the paper,
        without any approximations or optimizations. It may however be computationally expensive 
        for large datasets.
    
    Args:
        V_q (np.ndarray): Query neighborhood weight matrix (Nq x All)
        V_g (np.ndarray): Gallery neighborhood weight matrix (Ng x All)
        
    Returns:
        np.ndarray: Strict Jaccard distance matrix (Nq x Ng)
    """
    num_q = V_q.shape[0]
    num_g = V_g.shape[0]
    jaccard_dist = np.zeros((num_q, num_g), dtype=np.float32)

    for i in tqdm(range(num_q), desc="Computing Jaccard"):
        v_q_i = V_q[i]
        
        temp_min = np.minimum(v_q_i, V_g)
        temp_max = np.maximum(v_q_i, V_g)
        
        jaccard_dist[i] = 1.0 - (np.sum(temp_min, axis=1) / (np.sum(temp_max, axis=1) + 1e-12))
    return jaccard_dist


def compute_euclidean_distance(features1, features2):
    """
    Computes the squared Euclidean distance matrix between two sets of features.
    
    Args:
        features1 (np.ndarray): Shape (N1, D)
        features2 (np.ndarray): Shape (N2, D)
        
    Returns:
        np.ndarray: Matrix of shape (N1, N2) where [i, j] is ||f1_i - f2_j||^2
    """
    a2 = np.sum(np.square(features1), axis=1, keepdims=True)
    b2 = np.sum(np.square(features2), axis=1, keepdims=True).T
    dist = a2 + b2 - 2 * np.dot(features1, features2.T)
    
    return np.maximum(dist, 0.0)


def k_reciprocal_reranking(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3):
    """
    Re-ranks search results based on k-reciprocal nearest neighbors.
    
    Methodology based on: 
    Zhong, Z., Zheng, L., Cao, D., & Li, S. (2017). 'Re-ranking Person Re-identification 
    with k-Reciprocal Encoding.' Proceedings of the IEEE Conference on Computer 
    Vision and Pattern Recognition (CVPR).
    
    Args:
        q_feat (torch.Tensor): Query features (Nq x D).
        g_feat (torch.Tensor): Gallery features (Ng x D).
        k1 (int): Size of the neighborhood for reciprocity check.
        k2 (int): Size of the neighborhood for local expansion.
        lambda_value (float): Balancing factor between original and Jaccard distance.
        
    Returns:
        np.ndarray: Re-ranked distance matrix of shape (Nq, Ng).
    """
    q_feat = q_feat.cpu().numpy()
    g_feat = g_feat.cpu().numpy()
    m, n = q_feat.shape[0], g_feat.shape[0]

    all_feat = np.concatenate([q_feat, g_feat], axis=0)
    dist = compute_euclidean_distance(all_feat, all_feat)
    all_num = m + n

    initial_rank = np.argsort(dist, axis=1)

    # Build k-reciprocal neighborhood weight matrix V
    V = np.zeros((all_num, all_num), dtype=np.float32)
    for i in tqdm(range(all_num), desc="Building Neighborhoods"):

        # Find k-reciprocal neighbors
        forward_k1 = initial_rank[i, :k1 + 1]
        backward_k1 = initial_rank[forward_k1, :k1 + 1]
        reciprocal_indices = np.where(backward_k1 == i)[0]
        reciprocal_neighbors = forward_k1[reciprocal_indices]
        
        # Local expansion (k2)
        for candidate in reciprocal_neighbors:
            cand_forward = initial_rank[candidate, :int(np.around(k1/2)) + 1]
            cand_backward = initial_rank[cand_forward, :int(np.around(k1/2)) + 1]
            cand_recip_indices = np.where(cand_backward == candidate)[0]
            cand_recip_neighbors = cand_forward[cand_recip_indices]
            
            # Intersection threshold for expansion
            if len(np.intersect1d(reciprocal_neighbors, cand_recip_neighbors)) > 2/3 * len(cand_recip_neighbors):
                reciprocal_neighbors = np.append(reciprocal_neighbors, cand_forward[:k2])
        
        reciprocal_neighbors = np.unique(reciprocal_neighbors)
        
        # Gaussian-like weights for the neighborhood vector
        weight = np.exp(-dist[i, reciprocal_neighbors])
        V[i, reciprocal_neighbors] = weight / np.sum(weight)
    V_q = V[:m]
    V_g = V[m:]
    jaccard_dist = compute_jaccard_distance(V_q, V_g)

    original_dist = dist[:m, m:]
    original_dist = original_dist / np.max(original_dist)
    
    return lambda_value * jaccard_dist + (1 - lambda_value) * original_dist