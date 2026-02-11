"""
Market-1501 Dataset Loader module.
Handles image loading and identity (PID) mapping for Person Re-Identification.
"""

import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MarketDataset(Dataset):
    """
    Custom Dataset for Market-1501.
    Extracts PIDs and CamIDs from filenames: [PID]_c[CamID]s[SessionID]_[Frame].jpg
    """
    def __init__(self, root_dir, subset='train', transform=None):
        """
        Args:
            root_dir (str): Path to Market-1501 root folder.
            subset (str): One of 'train', 'test', or 'query'.
            transform (callable, optional): torchvision transforms to be applied.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Mapping subset to actual folder names
        subset_map = {
            'train': 'bounding_box_train',
            'test': 'bounding_box_test',
            'query': 'query'
        }
        sub_folder = subset_map.get(subset, subset)
        self.img_path = os.path.join(self.root_dir, sub_folder)
        
        self.files = sorted(glob.glob(os.path.join(self.img_path, "*.jpg")))
        
        # Filter out junk images (pid <= 0) and extract PIDs + CamIDs
        valid_files, pids, camids = [], [], []
        for f in self.files:
            name = os.path.basename(f)
            pid = int(name.split('_')[0])
            if pid > 0:
                camid = int(name.split('_')[1][1])  # e.g. 'c1s1_...' -> 1
                valid_files.append(f)
                pids.append(pid)
                camids.append(camid)
        
        self.files = valid_files
        self.pids = pids
        self.camids = camids
        
        # Re-map PIDs to continuous range [0, num_classes-1] for Cross-Entropy Loss
        unique_pids = sorted(list(set(self.pids)))
        self.pid_map = {pid: i for i, pid in enumerate(unique_pids)}

    def __len__(self):
        """Returns the total number of valid images."""
        return len(self.files)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the item to fetch.
        Returns:
            tuple: (image_tensor, mapped_label)
        """
        f = self.files[index]
        pid = self.pids[index]
        camid = self.camids[index]
        label = self.pid_map[pid]
        
        # Use PIL for compatibility with torchvision transforms
        img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label, camid