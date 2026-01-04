"""
ResNet-50 Architecture for Person Re-Identification.
Includes the BN-Neck (Bottleneck) trick to improve Re-ID performance.
"""

import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    """
    Modified ResNet-50 for Re-ID.
    Removes the final FC layer and adds a Batch Normalization bottleneck.
    """
    def __init__(self, num_classes, feature_dim=2048, last_stride=1):
        """
        Args:
            num_classes (int): Number of identities in the training set.
            feature_dim (int): Output dimension of the backbone (default 2048 for ResNet50).
            last_stride (int): If 1, removes downsampling in the last conv block to keep spatial resolution.
        """
        super(ResNet50, self).__init__()
        
        # Load backbone with ImageNet weights
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Re-ID Trick: Change stride of the last convolutional block (layer4)
        # Keeping higher spatial resolution (16x8 instead of 8x4) improves performance.
        if last_stride == 1:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
        
        # Remove original Fully Connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # BN-Neck (Bottleneck)
        # During training: Softmax Loss uses features AFTER BN. Triplet Loss uses features BEFORE BN.
        # During inference: Use features AFTER BN for cosine/euclidean distance.
        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)  # BN bias is redundant before a Linear layer
        self.bottleneck.apply(self._weights_init_kaiming)
        
        # Final classification layer
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def _weights_init_kaiming(self, m):
        """Initialize BN layer weights."""
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass.
        Returns:
            In training: (class_scores, raw_features)
            In inference: (normalized_features)
        """
        # Feature extraction
        v = self.backbone(x)
        v = v.view(v.size(0), -1)  # Flatten: [batch, 2048]
        
        # BNNeck logic
        f = self.bottleneck(v)
        
        if self.training:
            y = self.classifier(f)
            return y, v  # Return both for Cross-Entropy (y) and Triplet Loss (v)
        
        return f