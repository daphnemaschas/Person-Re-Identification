"""
Domain Adaptation Utilities for Person Re-Identification (WP6).

Provides tools for:
  - Embedding extraction and dimensionality reduction (t-SNE)
  - Domain gap visualization and statistical quantification
  - Systematic failure case identification and analysis
  - Image quality metric computation for diagnostic purposes

References:
  - Deng et al., "Image-Image Domain Adaptation with Preserved
    Self-Similarity and Domain-Dissimilarity", CVPR 2018
  - Zhong et al., "Generalizing a Person Retrieval Model Hetero-
    and Homogeneously", ECCV 2018
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_embeddings(model, loader, device):
    """
    Extract L2-normalized embeddings from a DataLoader.

    Args:
        model: Re-ID model (eval mode is set internally).
        loader: DataLoader yielding (images, labels, camids).
        device: Torch device.

    Returns:
        features: (N, D) numpy array of L2-normalized embeddings.
        pids: (N,) numpy array of person identity labels.
        camids: (N,) numpy array of camera IDs.
    """
    model.eval()
    all_feats, all_pids, all_camids = [], [], []

    with torch.no_grad():
        for imgs, labels, cams in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, p=2, dim=1)

            all_feats.append(feats.cpu().numpy())
            all_pids.append(labels.numpy())
            all_camids.append(cams.numpy())

    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_pids, axis=0),
        np.concatenate(all_camids, axis=0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality Reduction
# ──────────────────────────────────────────────────────────────────────────────

def compute_tsne(features, n_samples=None, perplexity=30, seed=42):
    """
    Compute 2D t-SNE projection of feature vectors.

    Args:
        features: (N, D) array of embeddings.
        n_samples: Subsample to this many points (None = use all).
        perplexity: t-SNE perplexity parameter.
        seed: Random seed for reproducibility.

    Returns:
        projections: (N', 2) array of 2D coordinates.
        indices: (N',) original indices of the selected samples.
    """
    rng = np.random.RandomState(seed)

    if n_samples is not None and n_samples < len(features):
        indices = rng.choice(len(features), size=n_samples, replace=False)
        features = features[indices]
    else:
        indices = np.arange(len(features))

    effective_perplexity = min(perplexity, len(features) // 4)
    effective_perplexity = max(effective_perplexity, 5)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=seed,
        init='pca',
        learning_rate='auto',
        n_iter=1000,
    )
    projections = tsne.fit_transform(features)

    return projections, indices


# ──────────────────────────────────────────────────────────────────────────────
# Domain Gap Visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_domain_gap(
    source_feats,
    target_feats,
    source_label="Market-1501 (Source)",
    target_label="Personal (Target)",
    n_samples=2000,
    perplexity=30,
    figsize=(10, 8),
    save_path=None,
    seed=42,
):
    """
    Visualize domain gap between source and target embeddings via t-SNE.

    Points from each domain are colored differently to reveal distribution
    shift between the training (source) and deployment (target) domains.

    Args:
        source_feats: (Ns, D) source domain embeddings.
        target_feats: (Nt, D) target domain embeddings.
        source_label: Legend label for source domain.
        target_label: Legend label for target domain.
        n_samples: Max points per domain for t-SNE (for speed).
        perplexity: t-SNE perplexity parameter.
        figsize: Matplotlib figure size.
        save_path: Optional path to save the figure.
        seed: Random seed.

    Returns:
        fig: Matplotlib figure.
    """
    rng = np.random.RandomState(seed)

    # Subsample each domain independently
    ns = min(n_samples, len(source_feats))
    nt = min(n_samples, len(target_feats))
    src_idx = rng.choice(len(source_feats), size=ns, replace=False)
    tgt_idx = rng.choice(len(target_feats), size=nt, replace=False)

    combined = np.concatenate(
        [source_feats[src_idx], target_feats[tgt_idx]], axis=0
    )
    domain_labels = np.array([0] * ns + [1] * nt)

    # t-SNE on combined embeddings
    effective_perp = min(perplexity, len(combined) // 4)
    effective_perp = max(effective_perp, 5)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perp,
        random_state=seed,
        init='pca',
        learning_rate='auto',
        n_iter=1000,
    )
    proj = tsne.fit_transform(combined)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    source_mask = domain_labels == 0
    target_mask = domain_labels == 1

    ax.scatter(
        proj[source_mask, 0], proj[source_mask, 1],
        c='#3498db', alpha=0.4, s=8, label=source_label,
    )
    ax.scatter(
        proj[target_mask, 0], proj[target_mask, 1],
        c='#e74c3c', alpha=0.4, s=8, label=target_label,
    )

    ax.set_title(
        "Domain Gap Visualization (t-SNE)", fontsize=14, fontweight='bold'
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Domain gap figure saved to %s", save_path)

    return fig


def visualize_identity_clusters(
    features,
    pids,
    domain_labels=None,
    n_identities=10,
    n_samples=1500,
    perplexity=30,
    figsize=(12, 8),
    save_path=None,
    seed=42,
):
    """
    Visualize t-SNE embedding space colored by person identity.

    When ``domain_labels`` is provided, uses different marker shapes
    (circles for source, triangles for target) to show how identity
    clusters differ across domains.

    Args:
        features: (N, D) embeddings.
        pids: (N,) identity labels (mapped).
        domain_labels: Optional (N,) array of 0 (source) / 1 (target).
        n_identities: Number of identities to display.
        n_samples: Max total points for t-SNE.
        perplexity: t-SNE perplexity.
        figsize: Matplotlib figure size.
        save_path: Optional save path.
        seed: Random seed.

    Returns:
        fig: Matplotlib figure.
    """
    rng = np.random.RandomState(seed)
    unique_pids = np.unique(pids)

    # Select identities with the most samples
    pid_counts = {pid: int(np.sum(pids == pid)) for pid in unique_pids}
    sorted_pids = sorted(
        pid_counts.keys(), key=lambda p: pid_counts[p], reverse=True
    )
    selected_pids = sorted_pids[:n_identities]

    # Gather indices for selected identities
    mask = np.isin(pids, selected_pids)
    sel_indices = np.where(mask)[0]

    if len(sel_indices) > n_samples:
        sel_indices = rng.choice(sel_indices, size=n_samples, replace=False)

    sel_feats = features[sel_indices]
    sel_pids = pids[sel_indices]

    # t-SNE
    projections, _ = compute_tsne(sel_feats, perplexity=perplexity, seed=seed)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.tab10 if n_identities <= 10 else plt.cm.tab20

    for i, pid in enumerate(selected_pids):
        pid_mask = sel_pids == pid
        color = cmap(i % cmap.N)

        if domain_labels is not None:
            sel_domains = domain_labels[sel_indices]
            src_mask = pid_mask & (sel_domains == 0)
            tgt_mask = pid_mask & (sel_domains == 1)
            ax.scatter(
                projections[src_mask, 0], projections[src_mask, 1],
                c=[color], marker='o', s=20, alpha=0.7,
            )
            ax.scatter(
                projections[tgt_mask, 0], projections[tgt_mask, 1],
                c=[color], marker='^', s=30, alpha=0.7,
            )
        else:
            ax.scatter(
                projections[pid_mask, 0], projections[pid_mask, 1],
                c=[color], s=20, alpha=0.7, label=f"ID {pid}",
            )

    ax.set_title(
        "Identity Clusters in Embedding Space (t-SNE)",
        fontsize=14, fontweight='bold',
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)

    if domain_labels is None and n_identities <= 15:
        ax.legend(fontsize=8, markerscale=2, ncol=2, loc='best')
    elif domain_labels is not None:
        legend_elements = [
            plt.Line2D(
                [0], [0], marker='o', color='gray',
                markerfacecolor='gray', markersize=8,
                linestyle='None', label='Source',
            ),
            plt.Line2D(
                [0], [0], marker='^', color='gray',
                markerfacecolor='gray', markersize=8,
                linestyle='None', label='Target',
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=11)

    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Domain Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_domain_statistics(feats_a, feats_b):
    """
    Compute statistical measures of domain gap between two embedding sets.

    Metrics:
      - Centroid distance (Euclidean) between mean embeddings.
      - Cosine similarity of centroids.
      - Intra-domain variance for each domain.
      - Linear MMD (Maximum Mean Discrepancy).

    Args:
        feats_a: (Na, D) embeddings from domain A.
        feats_b: (Nb, D) embeddings from domain B.

    Returns:
        dict with domain gap statistics.
    """
    mean_a = feats_a.mean(axis=0)
    mean_b = feats_b.mean(axis=0)

    # Centroid distance
    centroid_dist = float(np.linalg.norm(mean_a - mean_b))

    # Cosine similarity of centroids
    cos_sim = float(
        np.dot(mean_a, mean_b)
        / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-8)
    )

    # Intra-domain variance (mean squared distance to centroid)
    var_a = float(np.mean(np.sum((feats_a - mean_a) ** 2, axis=1)))
    var_b = float(np.mean(np.sum((feats_b - mean_b) ** 2, axis=1)))

    # Linear MMD
    mmd = _compute_mmd_linear(feats_a, feats_b)

    return {
        'centroid_distance': centroid_dist,
        'cosine_similarity': cos_sim,
        'variance_domain_a': var_a,
        'variance_domain_b': var_b,
        'mmd_linear': mmd,
    }


def _compute_mmd_linear(x, y, n_subsample=1000):
    """
    Compute linear MMD^2 between two sample sets.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 * E[k(x,y)]
    where k(a,b) = a^T b  (linear kernel).
    """
    rng = np.random.RandomState(42)

    if len(x) > n_subsample:
        x = x[rng.choice(len(x), n_subsample, replace=False)]
    if len(y) > n_subsample:
        y = y[rng.choice(len(y), n_subsample, replace=False)]

    xx = np.mean(x @ x.T)
    yy = np.mean(y @ y.T)
    xy = np.mean(x @ y.T)

    return float(xx + yy - 2 * xy)


# ──────────────────────────────────────────────────────────────────────────────
# Failure Case Analysis
# ──────────────────────────────────────────────────────────────────────────────

def identify_failure_cases(
    query_pids,
    gallery_pids,
    distmat,
    query_camids=None,
    gallery_camids=None,
    top_n=20,
):
    """
    Identify the hardest queries — those where the first correct match
    is ranked furthest from position 1.

    Args:
        query_pids: (Nq,) query identity labels.
        gallery_pids: (Ng,) gallery identity labels.
        distmat: (Nq, Ng) pairwise distance matrix.
        query_camids: Optional (Nq,) camera IDs (for same-camera exclusion).
        gallery_camids: Optional (Ng,) camera IDs.
        top_n: Number of worst queries to return.

    Returns:
        List of dicts sorted by difficulty (worst first), each containing:
          - query_idx, query_pid, first_correct_rank, ap, top_retrieved
    """
    indices = np.argsort(distmat, axis=1)
    failures = []

    for i in range(len(query_pids)):
        q_pid = query_pids[i]
        order = indices[i]

        # Validity mask: exclude same-PID + same-CamID (standard protocol)
        if query_camids is not None and gallery_camids is not None:
            q_camid = query_camids[i]
            remove = (gallery_pids[order] == q_pid) & (
                gallery_camids[order] == q_camid
            )
            keep = ~remove
        else:
            keep = np.ones(len(order), dtype=bool)

        valid_order = order[keep]
        matches = (gallery_pids[valid_order] == q_pid).astype(np.int32)

        if matches.sum() == 0:
            continue

        # First correct rank (1-indexed)
        first_correct = int(np.argmax(matches) + 1)

        # Average Precision
        num_relevant = matches.sum()
        precision_at_k = matches.cumsum() / (np.arange(len(matches)) + 1)
        ap = float((precision_at_k * matches).sum() / num_relevant)

        # Top-10 retrieved gallery images
        top_retrieved = []
        for j in range(min(10, len(valid_order))):
            g_idx = int(valid_order[j])
            top_retrieved.append({
                'gallery_idx': g_idx,
                'is_correct': bool(gallery_pids[g_idx] == q_pid),
                'distance': float(distmat[i, g_idx]),
            })

        failures.append({
            'query_idx': i,
            'query_pid': int(q_pid),
            'first_correct_rank': first_correct,
            'ap': ap,
            'top_retrieved': top_retrieved,
        })

    # Worst first (highest first_correct_rank)
    failures.sort(key=lambda x: x['first_correct_rank'], reverse=True)

    return failures[:top_n]


def visualize_failure_cases(
    failures,
    query_dataset,
    gallery_dataset,
    n_cases=5,
    top_k=5,
    figsize_per_case=(15, 3),
    save_path=None,
):
    """
    Visualize the worst failure cases: query + top-k retrieved gallery images.

    Correct matches are shown with green borders, incorrect with red.

    Args:
        failures: Output of ``identify_failure_cases()``.
        query_dataset: Dataset with ``.files`` and ``.pids`` attributes.
        gallery_dataset: Dataset with ``.files`` and ``.pids`` attributes.
        n_cases: Number of failure cases to show.
        top_k: Gallery images per query.
        figsize_per_case: (width, height) per row.
        save_path: Optional path to save the figure.

    Returns:
        fig: Matplotlib figure.
    """
    n_cases = min(n_cases, len(failures))
    if n_cases == 0:
        logger.warning("No failure cases to visualize.")
        return None

    fig, axes = plt.subplots(
        n_cases, top_k + 1,
        figsize=(figsize_per_case[0], figsize_per_case[1] * n_cases),
    )
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    for row, failure in enumerate(failures[:n_cases]):
        q_idx = failure['query_idx']
        q_path = query_dataset.files[q_idx]
        q_pid = failure['query_pid']

        # Query image
        q_img = Image.open(q_path).convert('RGB')
        axes[row, 0].imshow(q_img)
        axes[row, 0].set_title(
            f"Query\nID: {q_pid}\n1st match @ rank {failure['first_correct_rank']}",
            fontweight='bold', fontsize=9,
        )
        for spine in axes[row, 0].spines.values():
            spine.set_edgecolor('#2980b9')
            spine.set_linewidth(3)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Top-k gallery results
        for col, ret in enumerate(failure['top_retrieved'][:top_k]):
            g_idx = ret['gallery_idx']
            g_path = gallery_dataset.files[g_idx]
            g_pid = gallery_dataset.pids[g_idx]
            is_correct = ret['is_correct']

            g_img = Image.open(g_path).convert('RGB')
            axes[row, col + 1].imshow(g_img)

            color = '#27ae60' if is_correct else '#e74c3c'
            symbol = '\u2713' if is_correct else '\u2717'
            axes[row, col + 1].set_title(
                f"{symbol} Rank {col + 1}\nID: {g_pid}\nd={ret['distance']:.3f}",
                fontsize=8, color=color,
            )
            for spine in axes[row, col + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])

    plt.suptitle(
        "Failure Case Analysis \u2014 Hardest Queries",
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Image Quality Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_image_quality_metrics(dataset, indices=None):
    """
    Compute per-image quality metrics for failure pattern analysis.

    Metrics:
      - Resolution (height x width)
      - Brightness (mean pixel intensity of grayscale)
      - Contrast (std of grayscale pixel intensities)
      - Blur score (variance of Laplacian — lower = more blurry)

    Args:
        dataset: Dataset with ``get_raw_image(idx)`` method.
        indices: Optional subset of indices. Defaults to all.

    Returns:
        dict of numpy arrays keyed by metric name.
    """
    if indices is None:
        indices = list(range(len(dataset)))

    heights, widths = [], []
    brightness_scores, contrast_scores, blur_scores = [], [], []

    for idx in tqdm(indices, desc="Computing quality metrics", leave=False):
        try:
            img = dataset.get_raw_image(idx)
        except (FileNotFoundError, Exception):
            continue

        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brightness_scores.append(float(np.mean(gray)))
        contrast_scores.append(float(np.std(gray)))
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    return {
        'heights': np.array(heights),
        'widths': np.array(widths),
        'brightness': np.array(brightness_scores),
        'contrast': np.array(contrast_scores),
        'blur_score': np.array(blur_scores),
    }


def analyze_failure_patterns(
    failures,
    query_dataset,
    figsize=(16, 10),
    save_path=None,
):
    """
    Analyze systematic patterns in failure cases by comparing image quality
    metrics of failed queries against the overall query distribution.

    Produces a four-panel figure:
      1. Resolution distribution (all vs. failed)
      2. Brightness distribution
      3. Blur / sharpness distribution
      4. Failure severity vs. image brightness scatter

    Args:
        failures: Output of ``identify_failure_cases()``.
        query_dataset: Dataset with ``get_raw_image()`` method.
        figsize: Matplotlib figure size.
        save_path: Optional save path.

    Returns:
        fig: Matplotlib figure.
        metrics: dict with 'failures' and 'all' quality metrics.
    """
    failure_indices = [f['query_idx'] for f in failures]

    # Compute metrics for failures and a random sample of all queries
    rng = np.random.RandomState(42)
    n_all = min(len(query_dataset), 500)
    all_indices = rng.choice(len(query_dataset), size=n_all, replace=False).tolist()

    fail_metrics = compute_image_quality_metrics(query_dataset, failure_indices)
    all_metrics = compute_image_quality_metrics(query_dataset, all_indices)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Resolution
    ax = axes[0, 0]
    all_res = all_metrics['heights'] * all_metrics['widths']
    fail_res = fail_metrics['heights'] * fail_metrics['widths']
    ax.hist(all_res, bins=30, alpha=0.5, color='#3498db',
            label='All queries', density=True)
    if len(fail_res) > 0:
        ax.hist(fail_res, bins=15, alpha=0.6, color='#e74c3c',
                label='Failed queries', density=True)
    ax.set_xlabel('Resolution (pixels)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Image Resolution Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 2: Brightness
    ax = axes[0, 1]
    ax.hist(all_metrics['brightness'], bins=30, alpha=0.5, color='#3498db',
            label='All queries', density=True)
    if len(fail_metrics['brightness']) > 0:
        ax.hist(fail_metrics['brightness'], bins=15, alpha=0.6, color='#e74c3c',
                label='Failed queries', density=True)
    ax.set_xlabel('Mean Brightness', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Brightness Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 3: Blur / Sharpness
    ax = axes[1, 0]
    ax.hist(all_metrics['blur_score'], bins=30, alpha=0.5, color='#3498db',
            label='All queries', density=True)
    if len(fail_metrics['blur_score']) > 0:
        ax.hist(fail_metrics['blur_score'], bins=15, alpha=0.6, color='#e74c3c',
                label='Failed queries', density=True)
    ax.set_xlabel('Blur Score (Laplacian variance)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Sharpness Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 4: Scatter — failure severity vs brightness
    ax = axes[1, 1]
    if len(failures) > 0 and len(fail_metrics['brightness']) > 0:
        n_plot = min(len(failures), len(fail_metrics['brightness']))
        ranks = [f['first_correct_rank'] for f in failures[:n_plot]]
        ax.scatter(
            fail_metrics['brightness'][:n_plot], ranks,
            c='#e74c3c', s=40, alpha=0.7, edgecolors='black', linewidth=0.5,
        )
        ax.set_xlabel('Mean Brightness', fontsize=11)
        ax.set_ylabel('First Correct Rank', fontsize=11)
        ax.set_title('Failure Severity vs. Image Brightness', fontsize=12)
    else:
        ax.text(
            0.5, 0.5, "No failure data", ha='center', va='center',
            fontsize=12, transform=ax.transAxes,
        )
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Failure Pattern Analysis", fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, {'failures': fail_metrics, 'all': all_metrics}


# ──────────────────────────────────────────────────────────────────────────────
# Camera-based Domain Split (simulation when personal data is unavailable)
# ──────────────────────────────────────────────────────────────────────────────

def create_camera_domain_split(dataset, source_cams=(1, 2, 3), target_cams=(4, 5, 6)):
    """
    Create source/target domain indices from a Market-1501 dataset by camera ID.

    Useful for simulating cross-domain scenarios when a personal dataset
    is not yet available. Different camera views introduce viewpoint, lighting,
    and resolution differences that mimic domain shift.

    Args:
        dataset: MarketDataset instance with ``.camids`` attribute.
        source_cams: Tuple of camera IDs for the source domain.
        target_cams: Tuple of camera IDs for the target domain.

    Returns:
        source_indices: list of dataset indices belonging to source cameras.
        target_indices: list of dataset indices belonging to target cameras.
    """
    source_cams = set(source_cams)
    target_cams = set(target_cams)

    source_indices = [
        i for i, cam in enumerate(dataset.camids) if cam in source_cams
    ]
    target_indices = [
        i for i, cam in enumerate(dataset.camids) if cam in target_cams
    ]

    logger.info(
        "Camera split — Source cams %s: %d images | Target cams %s: %d images",
        source_cams, len(source_indices), target_cams, len(target_indices),
    )

    return source_indices, target_indices


def build_comparison_table(results_dict):
    """
    Print a formatted comparison table of Re-ID results across experiments.

    Args:
        results_dict: dict mapping experiment name → {'rank1': float, 'mAP': float, ...}.
                      Values are expected in [0, 1] scale (will be displayed as %).
    """
    header = f"{'Experiment':<35} {'Rank-1':>8} {'Rank-5':>8} {'mAP':>8}"
    print(header)
    print('\u2500' * len(header))

    for name, metrics in results_dict.items():
        r1 = metrics.get('rank1', 0) * 100
        r5 = metrics.get('rank5', 0) * 100
        mAP = metrics.get('mAP', 0) * 100
        print(f"{name:<35} {r1:>7.2f}% {r5:>7.2f}% {mAP:>7.2f}%")
