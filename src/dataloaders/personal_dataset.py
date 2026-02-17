"""
Personal Dataset Loader for Person Re-Identification.

Loads person crops from a directory structure produced by the YOLO detection
pipeline.  Mirrors the MarketDataset API so it can be used interchangeably
with the same training / evaluation code.

Supported directory layouts:

    Layout A — Label JSON (preferred):
        crops_dir/
            image_001_crop000.jpg
            image_001_crop001.jpg
            ...
        labels.json   # {"image_001_crop000.jpg": 0, ...}

    Layout B — Identity sub-folders:
        crops_dir/
            identity_0/
                crop_a.jpg
                crop_b.jpg
            identity_1/
                ...
"""

import os
import json
import glob
import logging
from pathlib import Path
from collections import Counter
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PersonalDataset(Dataset):
    """
    Dataset for personal smartphone photos after YOLO-based detection & cropping.

    Provides the same (image, label, camid) interface as MarketDataset
    for seamless integration with the existing evaluation and training pipeline.

    Args:
        root_dir: Path to the directory containing person crops.
        labels_path: Path to a JSON file mapping filenames → identity IDs.
                     If None, falls back to sub-folder layout where folder
                     names encode identity IDs.
        transform: Optional torchvision transforms applied to each image.
        default_camid: Camera ID assigned to all images (personal photos
                       typically lack camera metadata). Set to 0 by default.
    """

    def __init__(
        self,
        root_dir: str,
        labels_path: Optional[str] = None,
        transform=None,
        default_camid: int = 0,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.default_camid = default_camid

        self.files: list[str] = []
        self.pids: list[int] = []
        self.camids: list[int] = []

        if labels_path is not None and os.path.isfile(labels_path):
            self._load_from_json(labels_path)
        else:
            self._load_from_subfolders()

        if len(self.files) == 0:
            logger.warning("PersonalDataset: 0 images loaded from %s", root_dir)

        # Build contiguous label mapping (0 … N-1)
        unique_pids = sorted(set(self.pids))
        self.pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}

        logger.info(
            "PersonalDataset loaded: %d images, %d identities from %s",
            len(self.files), len(unique_pids), root_dir,
        )

    # ------------------------------------------------------------------
    # Dataset interface (matches MarketDataset)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.files)

    def __getitem__(self, index: int):
        """
        Fetch an image and its corresponding mapped label and camera ID.

        Args:
            index: Item index.

        Returns:
            tuple: (image_tensor, mapped_label, camid)
        """
        f = self.files[index]
        pid = self.pids[index]
        camid = self.camids[index]
        label = self.pid_map[pid]

        img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, camid

    def get_raw_image(self, index: int) -> np.ndarray:
        """
        Load a raw image as an RGB numpy array (matches MarketDataset API).

        Args:
            index: Item index.

        Returns:
            np.ndarray: RGB image (H, W, 3).
        """
        path = self.files[index]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Summary / EDA helpers
    # ------------------------------------------------------------------

    @property
    def num_identities(self) -> int:
        return len(self.pid_map)

    @property
    def num_images(self) -> int:
        return len(self.files)

    def identity_distribution(self) -> dict[int, int]:
        """Return {identity_id: image_count} for distribution analysis."""
        return dict(Counter(self.pids))

    def summary(self) -> dict:
        """Return dataset summary statistics for EDA."""
        dist = self.identity_distribution()
        counts = list(dist.values())
        return {
            "num_images": self.num_images,
            "num_identities": self.num_identities,
            "images_per_identity": {
                "min": min(counts) if counts else 0,
                "max": max(counts) if counts else 0,
                "mean": float(np.mean(counts)) if counts else 0.0,
                "median": float(np.median(counts)) if counts else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _load_from_json(self, labels_path: str):
        """Load images + labels from a JSON annotation file."""
        with open(labels_path, "r") as f:
            labels = json.load(f)

        for fname, pid in sorted(labels.items()):
            full_path = os.path.join(self.root_dir, fname)
            if os.path.isfile(full_path):
                self.files.append(full_path)
                self.pids.append(int(pid))
                self.camids.append(self.default_camid)
            else:
                logger.warning("File referenced in labels not found: %s", full_path)

    def _load_from_subfolders(self):
        """Load images from identity sub-folder structure."""
        root = Path(self.root_dir)
        if not root.is_dir():
            logger.warning("Root directory does not exist: %s", self.root_dir)
            return

        subdirs = sorted(
            d for d in root.iterdir() if d.is_dir()
        )

        if subdirs:
            # Sub-folder layout: each folder name is an identity
            for subdir in subdirs:
                try:
                    pid = int(subdir.name)
                except ValueError:
                    pid = hash(subdir.name) % (10**6)
                    logger.info(
                        "Non-integer folder name '%s' mapped to pid %d",
                        subdir.name, pid,
                    )

                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in self._IMAGE_EXTENSIONS:
                        self.files.append(str(img_path))
                        self.pids.append(pid)
                        self.camids.append(self.default_camid)
        else:
            # Flat directory — treat each image as its own identity (unlabeled)
            for img_path in sorted(root.iterdir()):
                if img_path.suffix.lower() in self._IMAGE_EXTENSIONS:
                    self.files.append(str(img_path))
                    self.pids.append(len(self.files) - 1)  # unique pseudo-id
                    self.camids.append(self.default_camid)
            if self.files:
                logger.warning(
                    "No labels found — each image assigned a unique pseudo-ID. "
                    "Provide a labels.json or use identity sub-folders for real labels."
                )
