"""
Visualization utilities for Re-ID results.
Helps interpret model predictions by showing Query vs Gallery matches.
"""

import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

def visualize_results(query_dataset, gallery_dataset, distmat, query_index=None, top_k=5):
    """
    Visualizes the top-k retrieved images for a specific or random query.
    
    Args:
        query_dataset (MarketDataset): Dataset containing query images.
        gallery_dataset (MarketDataset): Dataset containing gallery images.
        distmat (np.array): Precomputed distance matrix [num_query, num_gallery].
        query_index (int, optional): Index of the query to visualize. Random if None.
        top_k (int): Number of gallery results to show.
    """
    # 1. Pick a query
    if query_index is None:
        query_index = random.randint(0, len(query_dataset) - 1)
    
    query_img_path = query_dataset.files[query_index]
    query_pid = query_dataset.pids[query_index]
    
    # 2. Sort gallery indices by distance (closest first)
    # distmat[query_index] gives distances between this query and all gallery images
    sorted_indices = np.argsort(distmat[query_index])
    
    # 3. Plotting
    plt.figure(figsize=(15, 6))
    
    # Display Query Image
    query_img = Image.open(query_img_path).convert('RGB')
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img)
    plt.title(f"QUERY\nID: {query_pid}", fontweight='bold')
    plt.axis('off')
    
    # Display Top-K Gallery Results
    for i in range(top_k):
        gallery_idx = sorted_indices[i]
        gallery_img_path = gallery_dataset.files[gallery_idx]
        gallery_pid = gallery_dataset.pids[gallery_idx]
        
        img = Image.open(gallery_img_path).convert('RGB')
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(img)
        
        # Green if correct identity, Red otherwise
        is_correct = (gallery_pid == query_pid)
        color = 'green' if is_correct else 'red'
        
        plt.title(f"Rank {i+1}\nID: {gallery_pid}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()