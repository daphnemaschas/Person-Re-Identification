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
        sub_folder = "bounding_box_train" if subset == 'train' else "bounding_box_test"
        self.img_path = os.path.join(self.root_dir, sub_folder)
        
        self.files = sorted(glob.glob(os.path.join(self.img_path, "*.jpg")))
        
        self.pids = [int(os.path.basename(f).split('_')[0]) for f in self.files if int(os.path.basename(f).split('_')[0]) > 0]
        self.files = [f for f in self.files if int(os.path.basename(f).split('_')[0]) > 0]
        
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
        label = self.pid_map[pid]
        
        # Use PIL for compatibility with torchvision transforms
        img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label