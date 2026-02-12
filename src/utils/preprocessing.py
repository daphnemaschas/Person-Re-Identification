"""
Preprocessing and image transformation utilities for Re-ID analysis.
Supports color space conversion, filtering, and resolution scaling.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def adjust_illumination(image, brightness=1.0, contrast=1.0):
    """
    Simulates illumination changes by adjusting brightness and contrast.
    
    Args:
        image (PIL.Image): Input image.
        brightness (float): Factor to multiply pixel values (1.0 is neutral).
        contrast (float): Factor to adjust contrast.
        
    Returns:
        PIL.Image: Image with adjusted illumination.
    """
    img_np = np.array(image).astype(np.float32)
    res = img_np * contrast * brightness
    res = np.clip(res, 0, 255).astype(np.uint8)
    return Image.fromarray(res)


def get_channel_histograms(image_np, space='RGB'):
    """
    Computes histograms for each channel of the image.
    """
    hists = []
    for i in range(3):
        hist = cv2.calcHist([image_np], [i], None, [256], [0, 256])
        hists.append(hist)
    return hists


def convert_color_space(image, space='RGB'):
    """
    Converts a PIL image to different color spaces.
    
    Args:
        image (PIL.Image): Input image.
        space (str): Target color space ('RGB', 'HSV', 'LAB').
        
    Returns:
        np.array: Converted image in numpy format.
    """
    img_np = np.array(image)
    if space == 'HSV':
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    elif space == 'LAB':
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    return img_np


def apply_image_filter(image, filter_type='gaussian', kernel_size=5):
    """
    Applies spatial filters to a PIL image.
    
    Args:
        image (PIL.Image): Input image.
        filter_type (str): Type of filter ('gaussian', 'median', 'bilateral').
        kernel_size (int): Size of the filter kernel.
        
    Returns:
        PIL.Image: Filtered image.
    """
    img_np = np.array(image)
    if filter_type == 'gaussian':
        res = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        res = cv2.medianBlur(img_np, kernel_size)
    elif filter_type == 'bilateral':
        res = cv2.bilateralFilter(img_np, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    else:
        res = img_np
    return Image.fromarray(res)


def get_augmentation_pipeline(aug_type='none'):
    """
    Returns specific transformation pipelines for Re-ID.
    
    Args:
        aug_type (str): Type of augmentation ('standard', 'color_jitter', 'random_erasing').
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline.
    """
    base_transforms = [
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if aug_type == 'standard':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            *base_transforms
        ])
    elif aug_type == 'color_jitter':
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            *base_transforms
        ])
    elif aug_type == 'random_erasing':
        return transforms.Compose([
            *base_transforms,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    
    return transforms.Compose(base_transforms)
