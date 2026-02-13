"""
Market-1501 Dataset Loader module.
Handles image loading and identity (PID) mapping for Person Re-Identification.
"""

import os
import cv2
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
        
        subset_map = {
            'train': 'bounding_box_train',
            'test': 'bounding_box_test',
            'query': 'query'
        }
        sub_folder = subset_map.get(subset, subset)
        self.img_path = os.path.join(self.root_dir, sub_folder)
        
        self.files = sorted(glob.glob(os.path.join(self.img_path, "*.jpg")))
        
        valid_files, pids, camids = [], [], []
        for f in self.files:
            name = os.path.basename(f)
            pid = int(name.split('_')[0])
            if pid > 0:
                camid = int(name.split('_')[1][1])
                valid_files.append(f)
                pids.append(pid)
                camids.append(camid)
        
        self.files = valid_files
        self.pids = pids
        self.camids = camids
        
        unique_pids = sorted(list(set(self.pids)))
        self.pid_map = {pid: i for i, pid in enumerate(unique_pids)}

    def __len__(self):
        """Returns the total number of valid images."""
        return len(self.files)

    def __getitem__(self, index):
        """
        Fetches the image and its corresponding mapped label and CamID.

        Args:
            index (int): Index of the item to fetch.
        Returns:
            tuple: (image_tensor, mapped_label, camid)
        """
        f = self.files[index]
        pid = self.pids[index]
        camid = self.camids[index]
        label = self.pid_map[pid]
        
        img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label, camid
    
    def get_raw_image(self, index):
        """
        Loads a raw image using OpenCV and converts BGR to RGB.
        
        Args:
            index (int): Index of the item to fetch.   
        Returns:
            np.array: RGB image as a numpy array.
        """
        path = self.files[index]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)