import torch
import numpy as np
from tqdm import tqdm


def extract_features(model, loader, device):
    """Extract features, PIDs, and CamIDs from a loader."""
    model.eval()
    features, pids, camids = [], [], []
    
    with torch.no_grad():
        for imgs, labels, cams in tqdm(loader, desc="Extraction"):
            imgs = imgs.to(device)
            feat = model(imgs)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            
            features.append(feat.cpu())
            pids.extend(labels.numpy())
            camids.extend(cams.numpy())
            
    return torch.cat(features, 0), np.array(pids), np.array(camids)

def evaluate(query_feat, query_pids, gallery_feat, gallery_pids,
             query_camids=None, gallery_camids=None, distmat=None):
    """
    Calculates Rank-1 and mAP following the standard Market-1501 protocol.
    Gallery images with the same PID AND same CamID as the query are excluded
    from the ranking (they are trivially easy matches from the same camera view).
    """
    if distmat is None:
        distmat = compute_distmat(query_feat, gallery_feat)
    
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)  # Sort gallery by distance

    all_cmc = []
    aps = []

    for i in range(m):
        q_pid = query_pids[i]
        order = indices[i]  # Gallery indices sorted by distance
        
        # Build a validity mask: exclude gallery items with same PID + same CamID
        if query_camids is not None and gallery_camids is not None:
            q_camid = query_camids[i]
            # Invalid = same person AND same camera (trivial match)
            remove = (gallery_pids[order] == q_pid) & (gallery_camids[order] == q_camid)
            keep = ~remove
        else:
            keep = np.ones(n, dtype=bool)

        # Matches among the valid gallery entries
        raw_matches = (gallery_pids[order] == q_pid).astype(np.int32)
        matches = raw_matches[keep]

        if matches.sum() == 0:
            continue  # No valid match for this query

        # CMC (Cumulative Matching Characteristics)
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1  # Binary: found or not
        all_cmc.append(cmc[:50])  # Keep top-50 for CMC

        # AP (Average Precision)
        num_relevant = matches.sum()
        precision_at_k = matches.cumsum() / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / num_relevant
        aps.append(ap)

    all_cmc = np.array(all_cmc, dtype=np.float32).mean(axis=0)
    rank1 = all_cmc[0]
    rank5 = all_cmc[4] if len(all_cmc) > 4 else all_cmc[-1]
    rank10 = all_cmc[9] if len(all_cmc) > 9 else all_cmc[-1]
    mAP = np.mean(aps)

    print(f"  Rank-1: {rank1*100:.2f}% | Rank-5: {rank5*100:.2f}% | Rank-10: {rank10*100:.2f}% | mAP: {mAP*100:.2f}%")

    return rank1, mAP, distmat

def compute_distmat(query_feat, gallery_feat):
    """
    Computes Euclidean distance matrix between query and gallery features.
    distmat[i, j] = sqrt( ||q_i||^2 + ||g_j||^2 - 2 * q_i.T * g_j )
    """
    if torch.is_tensor(query_feat):
        query_feat = query_feat.detach().cpu().numpy()
    if torch.is_tensor(gallery_feat):
        gallery_feat = gallery_feat.detach().cpu().numpy()

    m, n = query_feat.shape[0], gallery_feat.shape[0]
    
    q2 = np.sum(np.square(query_feat), axis=1, keepdims=True) # (m, 1)
    g2 = np.sum(np.square(gallery_feat), axis=1, keepdims=True).T # (1, n)
    
    # a^2 + b^2 - 2ab
    distmat = q2 + g2 - 2 * np.dot(query_feat, gallery_feat.T)
    
    return np.sqrt(np.maximum(distmat, 0))