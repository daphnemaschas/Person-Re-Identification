"""
YOLO-based Person Detection Pipeline for Re-ID.

Detects persons in raw images using YOLOv11 and extracts cropped bounding boxes
suitable for downstream Re-ID feature extraction. Handles edge cases including
multiple detections per image, confidence filtering, minimum size constraints,
and aspect ratio validation.

Usage:
    detector = YOLOPersonDetector(model_name="yolo11n.pt", confidence=0.5)
    crops = detector.detect("path/to/image.jpg")
    detector.process_directory("raw_photos/", "output_crops/")
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single person detection within an image."""
    bbox: list[int]           # [x1, y1, x2, y2] in pixel coordinates
    confidence: float         # Detection confidence score
    crop: Optional[np.ndarray] = field(default=None, repr=False)  # RGB crop

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Height / Width — person crops are typically > 1.5."""
        return self.height / max(self.width, 1)


@dataclass
class ImageDetections:
    """All person detections for a single source image."""
    source_path: str
    image_height: int
    image_width: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def num_persons(self) -> int:
        return len(self.detections)


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class YOLOPersonDetector:
    """
    End-to-end person detection pipeline using YOLOv11.

    Responsibilities:
        1. Load a YOLO model (auto-downloads weights if needed).
        2. Run inference on single images or directories.
        3. Filter detections by confidence, minimum size, and aspect ratio.
        4. Extract person crops and persist them in Re-ID–ready format.

    Args:
        model_name: YOLO model weight file (e.g. "yolo11n.pt").
        confidence: Minimum detection confidence threshold.
        iou_threshold: IoU threshold for YOLO's internal NMS.
        min_height: Minimum crop height in pixels (filters tiny detections).
        min_width: Minimum crop width in pixels.
        min_aspect_ratio: Minimum H/W ratio (filters overly horizontal boxes).
        max_aspect_ratio: Maximum H/W ratio (filters overly vertical artifacts).
        padding_ratio: Fractional padding added around each detection box
                       to capture context (0.0 = no padding).
        device: Inference device ("cpu", "cuda", "mps", or None for auto).
    """

    # COCO class index for "person"
    _PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        min_height: int = 50,
        min_width: int = 25,
        min_aspect_ratio: float = 1.0,
        max_aspect_ratio: float = 6.0,
        padding_ratio: float = 0.05,
        device: Optional[str] = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install it with: pip install ultralytics"
            )

        # Handle model loading with auto-download support
        try:
            self.model = YOLO(model_name)
        except FileNotFoundError:
            # If exact path fails, try as a model name for auto-download
            model_base = os.path.splitext(os.path.basename(model_name))[0]
            logger.info(f"Model file not found, attempting auto-download: {model_base}")
            self.model = YOLO(f"{model_base}.pt")
        
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_height = min_height
        self.min_width = min_width
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.padding_ratio = padding_ratio
        self.device = device

        logger.info(
            "YOLOPersonDetector initialized | model=%s  conf=%.2f  iou=%.2f",
            model_name, confidence, iou_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> ImageDetections:
        """
        Detect persons in a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            ImageDetections object containing all valid person detections.
        """
        image_path = str(image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        raw_dets = self._run_inference(img_rgb)
        filtered = self._filter_detections(raw_dets, img_h=h, img_w=w)

        # Extract crops
        for det in filtered:
            x1, y1, x2, y2 = det.bbox
            det.crop = img_rgb[y1:y2, x1:x2].copy()

        result = ImageDetections(
            source_path=image_path,
            image_height=h,
            image_width=w,
            detections=filtered,
        )

        logger.info(
            "%s → %d raw detections, %d after filtering",
            os.path.basename(image_path), len(raw_dets), len(filtered),
        )
        return result

    def detect_from_array(self, img_rgb: np.ndarray) -> ImageDetections:
        """
        Detect persons from an in-memory RGB numpy array.

        Args:
            img_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            ImageDetections object.
        """
        h, w = img_rgb.shape[:2]
        raw_dets = self._run_inference(img_rgb)
        filtered = self._filter_detections(raw_dets, img_h=h, img_w=w)

        for det in filtered:
            x1, y1, x2, y2 = det.bbox
            det.crop = img_rgb[y1:y2, x1:x2].copy()

        return ImageDetections(
            source_path="<array>",
            image_height=h,
            image_width=w,
            detections=filtered,
        )

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        save_metadata: bool = True,
    ) -> dict:
        """
        Process all images in a directory: detect persons, save crops, and
        optionally write a metadata JSON file.

        Crop naming convention:
            {source_stem}_crop{idx:03d}.jpg

        Args:
            input_dir: Directory containing raw images.
            output_dir: Directory where cropped person images will be saved.
            extensions: Image file extensions to process.
            save_metadata: If True, saves a JSON file with detection metadata.

        Returns:
            Summary dict with counts and per-image statistics.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in extensions
        )
        if not image_files:
            logger.warning("No images found in %s", input_dir)
            return {"total_images": 0, "total_crops": 0, "per_image": []}

        all_metadata = []
        total_crops = 0

        for img_file in tqdm(image_files, desc="Detecting persons"):
            result = self.detect(str(img_file))
            img_meta = {
                "source": img_file.name,
                "image_size": [result.image_width, result.image_height],
                "num_detections": result.num_persons,
                "crops": [],
            }

            for idx, det in enumerate(result.detections):
                crop_name = f"{img_file.stem}_crop{idx:03d}.jpg"
                crop_path = output_path / crop_name

                # Save crop as RGB → BGR for cv2.imwrite
                cv2.imwrite(str(crop_path), cv2.cvtColor(det.crop, cv2.COLOR_RGB2BGR))

                img_meta["crops"].append({
                    "filename": crop_name,
                    "bbox": det.bbox,
                    "confidence": round(det.confidence, 4),
                    "size": [det.width, det.height],
                })
                total_crops += 1

            all_metadata.append(img_meta)

        # Write metadata
        if save_metadata:
            meta_path = output_path / "detection_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(all_metadata, f, indent=2)
            logger.info("Metadata saved to %s", meta_path)

        summary = {
            "total_images": len(image_files),
            "total_crops": total_crops,
            "per_image": all_metadata,
        }

        print(
            f"\nDetection complete: {len(image_files)} images → "
            f"{total_crops} person crops saved to {output_dir}"
        )
        return summary

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def visualize_detections(
        self,
        image_path: str,
        result: Optional[ImageDetections] = None,
        figsize: tuple[int, int] = (14, 8),
        show_crops: bool = True,
    ):
        """
        Draw bounding boxes on the source image and optionally display crops.

        Args:
            image_path: Path to the original image.
            result: Pre-computed ImageDetections (runs detection if None).
            figsize: Matplotlib figure size.
            show_crops: Whether to show individual crops below.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if result is None:
            result = self.detect(image_path)

        img = Image.open(image_path).convert("RGB")

        n_crops = result.num_persons if show_crops else 0
        n_rows = 2 if n_crops > 0 else 1
        n_cols = max(n_crops, 1)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1] if n_rows == 2 else [1]},
        )

        # Top row: full image with bounding boxes
        if n_rows == 1 and n_cols == 1:
            ax_img = axes
        elif n_rows == 2:
            ax_img = fig.add_subplot(n_rows, 1, 1)
            # Hide the top row's original axes
            if n_cols > 1:
                for a in axes[0]:
                    a.axis("off")
            else:
                axes[0].axis("off")
        else:
            ax_img = axes[0] if n_cols > 1 else axes

        # Redraw with a spanning subplot
        fig.clf()
        if n_rows == 2:
            gs = fig.add_gridspec(2, max(n_cols, 1), height_ratios=[3, 1], hspace=0.3)
            ax_img = fig.add_subplot(gs[0, :])
        else:
            ax_img = fig.add_subplot(1, 1, 1)

        ax_img.imshow(np.array(img))
        ax_img.set_title(
            f"{os.path.basename(image_path)}  —  {result.num_persons} person(s) detected",
            fontsize=12, fontweight="bold",
        )
        ax_img.axis("off")

        colors = plt.cm.Set2(np.linspace(0, 1, max(result.num_persons, 1)))

        for i, det in enumerate(result.detections):
            x1, y1, x2, y2 = det.bbox
            rect = patches.Rectangle(
                (x1, y1), det.width, det.height,
                linewidth=2, edgecolor=colors[i], facecolor="none",
            )
            ax_img.add_patch(rect)
            ax_img.text(
                x1, y1 - 4,
                f"Person {i+1} ({det.confidence:.2f})",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.8),
            )

        # Bottom row: individual crops
        if show_crops and n_crops > 0:
            for i, det in enumerate(result.detections):
                ax_crop = fig.add_subplot(gs[1, i])
                ax_crop.imshow(det.crop)
                ax_crop.set_title(
                    f"Crop {i+1}\n{det.width}×{det.height} | conf={det.confidence:.2f}",
                    fontsize=9,
                )
                ax_crop.axis("off")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _run_inference(self, img_rgb: np.ndarray) -> list[Detection]:
        """Run YOLO inference and return raw person detections."""
        results = self.model(
            img_rgb,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[self._PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(Detection(bbox=xyxy, confidence=conf))

        return detections

    def _filter_detections(
        self,
        detections: list[Detection],
        img_h: int,
        img_w: int,
    ) -> list[Detection]:
        """
        Apply post-processing filters on raw detections.

        Filters:
            1. Minimum dimensions (height, width).
            2. Aspect ratio range (H/W).
            3. Padding & boundary clamping.
        """
        filtered = []
        for det in detections:
            # Size filtering
            if det.height < self.min_height or det.width < self.min_width:
                logger.debug(
                    "Filtered out small detection: %dx%d (min %dx%d)",
                    det.width, det.height, self.min_width, self.min_height,
                )
                continue

            # Aspect ratio filtering
            ar = det.aspect_ratio
            if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
                logger.debug(
                    "Filtered out detection with aspect ratio %.2f (range [%.1f, %.1f])",
                    ar, self.min_aspect_ratio, self.max_aspect_ratio,
                )
                continue

            # Apply padding and clamp to image boundaries
            padded_bbox = self._apply_padding(det.bbox, img_h, img_w)
            det.bbox = padded_bbox
            filtered.append(det)

        # Sort by confidence descending
        filtered.sort(key=lambda d: d.confidence, reverse=True)
        return filtered

    def _apply_padding(
        self, bbox: list[int], img_h: int, img_w: int
    ) -> list[int]:
        """Expand bounding box by padding_ratio and clamp to image boundaries."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)
        return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Utility functions (standalone usage)
# ---------------------------------------------------------------------------

def create_dataset_split(
    crops_dir: str,
    labels_path: str,
    output_dir: str,
    split_ratio: float = 0.5,
    strategy: str = "identity_disjoint",
    seed: int = 42,
) -> dict:
    """
    Split annotated person crops into two subsets for domain adaptation experiments.

    Supports two strategies:
        - 'identity_disjoint': Each identity appears in exactly one split.
          Best for evaluating generalization to unseen identities.
        - 'proportional': Each identity is proportionally divided across splits.
          Useful when identity count is very small.

    Args:
        crops_dir: Directory containing cropped person images.
        labels_path: Path to the annotation JSON file with format:
                     {"crop_filename.jpg": identity_id, ...}
        output_dir: Base output directory. Creates 'part1/' and 'part2/' inside.
        split_ratio: Fraction of data (or identities) assigned to part1.
        strategy: One of 'identity_disjoint' or 'proportional'.
        seed: Random seed for reproducibility.

    Returns:
        Dict with split statistics.
    """
    import shutil
    from collections import defaultdict

    rng = np.random.RandomState(seed)

    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Group filenames by identity
    id_to_files = defaultdict(list)
    for fname, identity in labels.items():
        id_to_files[identity].append(fname)

    all_ids = sorted(id_to_files.keys())
    n_ids = len(all_ids)

    part1_dir = Path(output_dir) / "personal_part1"
    part2_dir = Path(output_dir) / "personal_part2"
    part1_dir.mkdir(parents=True, exist_ok=True)
    part2_dir.mkdir(parents=True, exist_ok=True)

    part1_files, part2_files = [], []

    if strategy == "identity_disjoint":
        shuffled = rng.permutation(all_ids).tolist()
        n_part1 = max(1, int(n_ids * split_ratio))
        part1_ids = set(shuffled[:n_part1])
        part2_ids = set(shuffled[n_part1:])

        for pid in part1_ids:
            for fname in id_to_files[pid]:
                src = Path(crops_dir) / fname
                if src.exists():
                    shutil.copy2(str(src), str(part1_dir / fname))
                    part1_files.append(fname)

        for pid in part2_ids:
            for fname in id_to_files[pid]:
                src = Path(crops_dir) / fname
                if src.exists():
                    shutil.copy2(str(src), str(part2_dir / fname))
                    part2_files.append(fname)

    elif strategy == "proportional":
        for pid in all_ids:
            files = id_to_files[pid]
            rng.shuffle(files)
            n_part1 = max(1, int(len(files) * split_ratio))
            for fname in files[:n_part1]:
                src = Path(crops_dir) / fname
                if src.exists():
                    shutil.copy2(str(src), str(part1_dir / fname))
                    part1_files.append(fname)
            for fname in files[n_part1:]:
                src = Path(crops_dir) / fname
                if src.exists():
                    shutil.copy2(str(src), str(part2_dir / fname))
                    part2_files.append(fname)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'identity_disjoint' or 'proportional'.")

    # Save split-specific label files
    for split_name, split_files, split_dir in [
        ("part1", part1_files, part1_dir),
        ("part2", part2_files, part2_dir),
    ]:
        split_labels = {f: labels[f] for f in split_files}
        with open(split_dir / "labels.json", "w") as f:
            json.dump(split_labels, f, indent=2)

    stats = {
        "strategy": strategy,
        "total_identities": n_ids,
        "total_images": len(labels),
        "part1_identities": len(set(labels[f] for f in part1_files)),
        "part1_images": len(part1_files),
        "part2_identities": len(set(labels[f] for f in part2_files)) if part2_files else 0,
        "part2_images": len(part2_files),
    }

    print(f"\nDataset split ({strategy}):")
    print(f"  Part 1: {stats['part1_images']} images, {stats['part1_identities']} identities → {part1_dir}")
    print(f"  Part 2: {stats['part2_images']} images, {stats['part2_identities']} identities → {part2_dir}")
    return stats
