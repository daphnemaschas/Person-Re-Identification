import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.functional.retrieval import retrieval_precision


def extract_features(model, loader, device):
    """Extract features from a loader and return them as tensors."""
    model.eval()
    features, pids, camids = [], [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extraction"):
            imgs = imgs.to(device)
            # En mode eval, notre modèle ResNet50ReID ne renvoie que les features après le BN-Neck
            feat = model(imgs)
            
            # On normalise les features (L2 norm) pour faciliter le calcul de distance
            feat = torch.nn.functional.normalize(feat, p_2, dim=1)
            
            features.append(feat.cpu())
            pids.extend(labels.numpy())
            
    return torch.cat(features, 0), np.array(pids)

def evaluate(query_feat, query_pids, gallery_feat, gallery_pids):
    """Calculates Rank-1 and mAP."""
    # Matrix of distance (Euclidian): Dist(A,B) = sqrt(A^2 + B^2 - 2AB)
    distmat = compute_distmat(query_feat, gallery_feat)

    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1) # Trier par distance
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)
    
    # Rank-1 : First match
    rank1 = matches[:, 0].mean()
    
    # mAP simplified
    aps = []
    for i in range(m):
        # Only when PID is the same
        relevant_indices = np.where(matches[i] == 1)[0]
        if len(relevant_indices) == 0: 
            continue
        
        ap = 0
        for j, pos in enumerate(relevant_indices):
            precision = (j + 1) / (pos + 1)
            ap += precision
        aps.append(ap / len(relevant_indices))
        
    return rank1, np.mean(aps), distmat

def compute_distmat(query_feat, gallery_feat):
    """
    Computes Euclidean distance matrix between query and gallery features.
    distmat[i, j] = sqrt( ||q_i||^2 + ||g_j||^2 - 2 * q_i.T * g_j )
    """
    m, n = query_feat.size(0), gallery_feat.size(0)
    
    # a^2 + b^2
    distmat = torch.pow(query_feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gallery_feat, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    
    # - 2ab
    distmat.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
    
    # We clamp to 0 to avoid negative micro-values due to imprecision
    return distmat.clamp(min=0).sqrt().cpu().numpy()