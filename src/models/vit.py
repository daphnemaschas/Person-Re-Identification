"""
Vision Transformer (ViT-Base) Architecture for Person Re-Identification.
Mirrors the BN-Neck design from ResNet50 for consistent training and evaluation.

Architecture overview:
    ViT-Base/16 (ImageNet pretrained) → [CLS] token (768-d)
    → BN-Neck (BatchNorm1d) → Classifier (Linear)

Training:  returns (class_scores, raw_features)
Inference: returns normalized BN-Neck features
"""

import torch
import torch.nn as nn
from torchvision import models


class ViTReID(nn.Module):
    """
    ViT-Base/16 adapted for Person Re-Identification with BN-Neck.

    The [CLS] token output serves as the global image representation,
    analogous to the GAP output in ResNet-50.
    """

    # Layer group names for interpretability and freezing control
    LAYER_GROUPS = {
        'patch_embed': ['conv_proj', 'class_token', 'encoder.pos_embedding'],
        'encoder_early': ['encoder.layers.encoder_layer_0',
                          'encoder.layers.encoder_layer_1',
                          'encoder.layers.encoder_layer_2',
                          'encoder.layers.encoder_layer_3',
                          'encoder.layers.encoder_layer_4',
                          'encoder.layers.encoder_layer_5'],
        'encoder_late':  ['encoder.layers.encoder_layer_6',
                          'encoder.layers.encoder_layer_7',
                          'encoder.layers.encoder_layer_8',
                          'encoder.layers.encoder_layer_9',
                          'encoder.layers.encoder_layer_10',
                          'encoder.layers.encoder_layer_11'],
        'encoder_ln':    ['encoder.ln'],
    }

    def __init__(self, num_classes: int, feature_dim: int = 768):
        """
        Args:
            num_classes: Number of identities in the training set.
            feature_dim: Output dimension of the ViT backbone (768 for ViT-Base).
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Backbone: ViT-Base/16 pretrained on ImageNet-1K
        # ------------------------------------------------------------------
        vit = models.vit_b_16(weights='IMAGENET1K_V1')

        # Remove the original classification head (vit.heads)
        vit.heads = nn.Identity()
        self.backbone = vit

        # ------------------------------------------------------------------
        # BN-Neck  (same design as ResNet50 Re-ID)
        # ------------------------------------------------------------------
        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(self._weights_init_kaiming)

        # ------------------------------------------------------------------
        # Classifier
        # ------------------------------------------------------------------
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        self._feature_dim = feature_dim

    # ----- Initialization helpers -----------------------------------------

    @staticmethod
    def _weights_init_kaiming(m):
        """Initialize BN layer weights (same convention as ResNet50 model)."""
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    # ----- Freezing strategies --------------------------------------------

    def freeze_backbone(self, mode: str = 'full'):
        """
        Control which parts of the backbone are frozen.

        Args:
            mode:
                - 'feature_extraction': Freeze entire backbone; train BN-Neck + classifier only.
                - 'partial': Freeze patch embedding + first 6 encoder layers;
                             train last 6 layers + LayerNorm + head.
                - 'full': Unfreeze everything (full fine-tuning).
        """
        if mode == 'feature_extraction':
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif mode == 'partial':
            # Freeze everything first, then selectively unfreeze
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze late encoder layers (6-11) and final LayerNorm
            for name, param in self.backbone.named_parameters():
                for group_key in ('encoder_late', 'encoder_ln'):
                    if any(prefix in name for prefix in self.LAYER_GROUPS[group_key]):
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

        # Report
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Freeze mode: '{mode}' | Trainable: {trainable:,} / {total:,} params "
              f"({100 * trainable / total:.1f}%)")

    # ----- Forward --------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input images [B, 3, H, W].  ViT-Base/16 expects 224×224 but
               torchvision's implementation interpolates positional embeddings
               automatically for other resolutions.

        Returns:
            Training:  (class_scores, raw_features)
            Inference: BN-normalised features for distance computation
        """
        v = self.backbone(x)            # [B, feature_dim]
        f = self.bottleneck(v)           # BN-Neck

        if self.training:
            y = self.classifier(f)
            return y, v                  # Cross-Entropy on y, Triplet on v

        return f
