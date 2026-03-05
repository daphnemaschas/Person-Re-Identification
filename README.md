# Person Re-Identification

A person re-identification (Re-ID) system designed to link images of an individual taken from different angles and under different lighting conditions. It targets two use-cases: **video surveillance tracking** and **automatic photo album clustering**.

> **Dual course project** — Visual Recognition (Prof. Hudelot) & Deep Learning (Prof. Le Borgne), CentraleSupélec 2025-2026.

---

## Datasets

| Dataset | Description | Usage |
|---|---|---|
| [Market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501) | 32,668 images, 1,501 identities, 6 cameras | Supervised training & academic benchmark |
| Personal Dataset | Smartphone photos, raw scenes, 10 identities, 3 cameras| Domain adaptation & "in-the-wild" evaluation |

---

## Project Structure

```
├── config.yaml                  # All hyperparameters and paths
├── requirements.txt
├── notebooks/
│   ├── eda.ipynb                        # EDA on Market-1501
│   ├── market_1501_resnet50.ipynb       # ResNet-50 baseline training
│   ├── ablation_freezing.ipynb          # WP4 — Freezing strategies ablation
│   ├── preprocessing_analysis.ipynb     # WP1 — Image preprocessing study
│   ├── classical_vs_deep_features.ipynb # WP2 — Feature comparison
│   ├── market_1501_vit.ipynb            # WP5 — ViT training
│   ├── architecture_comparison.ipynb    # WP5 — CNN vs Transformer comparison
│   ├── detection_pipeline.ipynb         # WP3 — YOLO detection demo
│   ├── domain_adaptation.ipynb          # WP6 — Zero-shot & transfer experiments
│   └── clustering_albums.ipynb          # WP7 — Album clustering application
├── src/
│   ├── dataloaders/
│   │   ├── market_dataset.py            # Market-1501 dataloader
│   │   └── personal_dataset.py          # WP3 — Personal dataset loader
│   ├── models/
│   │   ├── resnet50.py                  # ResNet-50 + BN-Neck for Re-ID
│   │   └── vit.py                       # WP5 — ViT-Base for Re-ID
│   ├── detection/
│   │   └── yolo_pipeline.py             # WP3 — YOLO person detection
│   └── utils/
│       ├── evaluator.py                 # Rank-1, mAP, CMC evaluation
│       ├── losses.py                    # Triplet loss with hard mining
│       ├── trainer.py                   # Training loop with history
│       ├── visualizer.py                # Query vs gallery visualization
│       ├── logging.py                   # Checkpoint save/load
│       ├── preprocessing.py             # WP1 — Filtering, augmentation utils
│       ├── classical_features.py        # WP2 — SIFT/HOG/color histograms
│       ├── reranking.py                 # WP2 — Re-ranking post-processing
│       └── clustering.py                # WP7 — Identity clustering
└── results/
```

---

## Work Packages & Task Assignment

### Coverage Matrix

Each WP is tagged with its primary course relevance:
- 🔬 **VR** = Visual Recognition (image processing, feature extraction, classical CV, geometric analysis)
- 🧠 **DL** = Deep Learning (neural architectures, training strategies, transfer learning)

---

### WP1 — Image Preprocessing & Augmentation Analysis 🔬

> *Visual Recognition: filtering, color spaces, image transformations (Cours 1-3)*

Study how preprocessing choices impact Re-ID performance.

| # | Sub-task | Description |
|---|---|---|
| 1.1 | **Color space analysis** | Compare Re-ID feature distributions across RGB, HSV, Lab color spaces. Visualize how illumination changes affect each space (relates to VR Cours 2). |
| 1.2 | **Filtering study** | Apply Gaussian, median, bilateral filters to Market-1501 images. Measure impact on feature quality and Re-ID metrics (relates to VR Cours 3). |
| 1.3 | **Data augmentation impact** | Implement Re-ID-specific augmentations (random erasing, color jitter, horizontal flip) and measure their effect on Rank-1/mAP. |
| 1.4 | **Resolution analysis** | Downsample images to various resolutions, measure Re-ID degradation — relevant to real-world camera variability. |

**Deliverable**: `notebooks/preprocessing_analysis.ipynb` + `src/utils/preprocessing.py`

---

### WP2 — Classical vs Deep Feature Analysis 🔬🧠

> *Visual Recognition: feature extraction, description, matching (Cours 4-5)*
> *Deep Learning: CNN feature understanding (Cours 3)*

Compare classical Computer Vision descriptors against deep embeddings for person matching.

| # | Sub-task | Description |
|---|---|---|
| 2.1 | **Classical descriptors** | Extract SIFT keypoints, HOG descriptors, and color histograms from Market-1501 crops. Implement in `src/utils/classical_features.py`. |
| 2.2 | **Feature matching** | Use BFMatcher / FLANN to match SIFT features between query and gallery. Compute Rank-1/mAP with classical features. |
| 2.3 | **Deep embeddings visualization** | Extract ResNet-50 embeddings, visualize with t-SNE/UMAP, color by identity. Show how the embedding space separates identities. |
| 2.4 | **Quantitative comparison** | Side-by-side table: SIFT vs HOG vs Color Histogram vs ResNet-50 vs ViT on Rank-1/mAP/speed. |
| 2.5 | **Re-ranking** | Implement k-reciprocal re-ranking (a post-processing algorithm) in `src/utils/reranking.py`. Measure improvement on deep features. |

**Deliverable**: `notebooks/classical_vs_deep_features.ipynb` + `src/utils/classical_features.py` + `src/utils/reranking.py`

---

### WP3 — Personal Dataset & YOLO Detection Pipeline 🔬

> *Visual Recognition: detection, segmentation, real-world pipeline (Cours 1, 6)*

Build the end-to-end pipeline from raw smartphone photos to labeled Re-ID crops.

| # | Sub-task | Description |
|---|---|---|
| 3.1 | **YOLO detection pipeline** | Implement `src/detection/yolo_pipeline.py`: load YOLOv11, detect persons in raw photos, extract crops with confidence scores. |
| 3.2 | **Detection edge cases** | Handle multiple persons per image, overlapping bboxes (NMS), minimum size filtering, confidence thresholds. |
| 3.3 | **Photo collection & annotation** | Collect smartphone photos from team members. Manually annotate identity labels on the crops. |
| 3.4 | **Dataset split** | Divide into `personal_part1` (fine-tuning) and `personal_part2` (testing) with identity-disjoint or proportional splits. |
| 3.5 | **PersonalDataset class** | Create `src/dataloaders/personal_dataset.py` with same API as `MarketDataset`. |
| 3.6 | **EDA** | Distribution analysis of the personal dataset (identities, cameras, image quality). |

**Deliverable**: `notebooks/detection_pipeline.ipynb` + `src/detection/yolo_pipeline.py` + `src/dataloaders/personal_dataset.py`

---

### WP4 — Freezing Ablation Study 🧠

> *Deep Learning: transfer learning, fine-tuning strategies (Cours 3, TD3)*

Compare 3 backbone freezing strategies on ResNet-50.

| # | Sub-task | Description |
|---|---|---|
| 4.1 | **Feature extraction** | Freeze entire backbone, train only BN-Neck + classifier. |
| 4.2 | **Partial fine-tuning** | Freeze conv1→layer2, train layer3/4 + head. |
| 4.3 | **Full fine-tuning** | Train the entire network. |
| 4.4 | **Comparison plots** | Loss curves, Rank-1/mAP bar charts, trainable parameter counts. |

**Deliverable**: `notebooks/ablation_freezing.ipynb` *(already implemented)*

---

### WP5 — Architecture Comparison (CNN vs Transformers) 🧠

> *Deep Learning: CNN, Transformers, architecture design (Cours 3-4)*

| # | Sub-task | Description |
|---|---|---|
| 5.1 | **ViT-Base Re-ID model** | Implement `src/models/vit.py` with a ViT-Base backbone + Re-ID head (same BN-Neck pattern). |
| 5.2 | **Training notebook** | Create `notebooks/market_1501_vit.ipynb` — train & evaluate on Market-1501. |
| 5.3 | **Computational cost analysis** | Compare FLOPs, parameter count, inference time (ms/image) between ResNet-50 and ViT. |
| 5.4 | **Comparison notebook** | `notebooks/architecture_comparison.ipynb` — Rank-1/mAP/cost table, CMC curves side by side. |

**Deliverable**: `src/models/vit.py` + `notebooks/architecture_comparison.ipynb`

---

### WP6 — Domain Adaptation Experiments 🔬🧠

> *Visual Recognition: recognition under domain shift (Cours 5)*
> *Deep Learning: transfer learning, generalization (Cours 3, 5)*

| # | Sub-task | Description |
|---|---|---|
| 6.1 | **Exp A — Zero-shot** | Load Market-1501-trained model → evaluate directly on personal dataset. Analyze failure modes. |
| 6.2 | **Exp B — Transfer** | Fine-tune on `personal_part1` → evaluate on `personal_part2`. Compare with zero-shot. |
| 6.3 | **Domain gap analysis** | Visualize embedding distributions (t-SNE) of Market-1501 vs personal data — show the domain shift. |
| 6.4 | **Failure case study** | Qualitative analysis: which types of images fail? (clothing, resolution, occlusion, background). |

**Depends on**: WP3
**Deliverable**: `notebooks/domain_adaptation.ipynb`

---

### WP7 — Clustering Application (Album Indexing) 🔬

> *Visual Recognition: recognition, segmentation, unsupervised grouping (Cours 5, Segmentation)*

| # | Sub-task | Description |
|---|---|---|
| 7.1 | **Clustering pipeline** | Extract embeddings from all person crops, cluster with DBSCAN or Agglomerative Clustering on cosine distance. |
| 7.2 | **Cluster evaluation** | Evaluate with NMI, ARI, Purity (using ground-truth labels for validation only). |
| 7.3 | **Threshold sensitivity** | Study the impact of distance threshold on clustering quality (precision/recall of grouping). |
| 7.4 | **Album demo** | Visualize clusters as a photo grid — "Person A: 12 photos, Person B: 8 photos...". |

**Deliverable**: `notebooks/clustering_albums.ipynb` + `src/utils/clustering.py`

---

## Course Coverage Summary

| Visual Recognition Topic | Covered in |
|---|---|
| Color spaces, illumination | WP1.1 |
| Spatial filtering | WP1.2 |
| Feature detection (keypoints) | WP2.1, WP2.2 |
| Feature description (SIFT, HOG) | WP2.1, WP2.2 |
| Feature matching | WP2.2 |
| Detection | WP3.1, WP3.2 |
| Segmentation / Grouping | WP7.1 |
| Recognition | All WPs (core task) |

| Deep Learning Topic | Covered in |
|---|---|
| CNN architecture | ResNet-50 baseline, WP5 |
| Transformers | WP5 (ViT) |
| Transfer learning | WP4, WP6 |
| Regularization / Generalization | WP4 (freezing), WP1.3 (augmentation) |
| Loss functions | Triplet + Cross-Entropy (baseline) |
| Training strategies (optimizers, schedulers) | All training notebooks |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Market-1501
pip install kaggle
kaggle datasets download -d pengcw1/market-1501 -p data/
unzip data/market-1501.zip -d data/

# 3. Run the baseline
# Open notebooks/market_1501_resnet50.ipynb and run all cells
```
