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

    def freeze_backbone(self, mode='full'):
        """
        Control which parts of the backbone are frozen for ablation study.
        
        Args:
            mode (str): Freezing strategy.
                - 'feature_extraction': Freeze entire backbone, train only BN-Neck + classifier.
                - 'partial': Freeze early layers (conv1 → layer2), train layer3 + layer4 + head.
                - 'full': Unfreeze everything (full fine-tuning).
        """
        if mode == 'feature_extraction':
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        elif mode == 'partial':
            # backbone is nn.Sequential of resnet children (minus last FC):
            # [0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool, [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4, [8]=avgpool
            for i, child in enumerate(self.backbone.children()):
                if i <= 5:  # Freeze conv1, bn1, relu, maxpool, layer1, layer2
                    for param in child.parameters():
                        param.requires_grad = False
                else:  # Unfreeze layer3, layer4, avgpool
                    for param in child.parameters():
                        param.requires_grad = True
                        
        elif mode == 'full':
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze mode: {mode}")
        
        # BN-Neck and classifier are always trainable
        for param in self.bottleneck.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Report trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Freeze mode: '{mode}' | Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

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