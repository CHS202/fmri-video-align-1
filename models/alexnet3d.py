'''AlexNet3D in PyTorch.

3D extension of AlexNet (Krizhevsky et al., 2012) for video classification.
All Conv2d layers are replaced with Conv3d, and BatchNorm is added after each
conv for training stability.

Pretrained 3D AlexNet weights do NOT exist in the public domain (no Kinetics /
Sports-1M checkpoint has been released for AlexNet-style architectures).
We therefore fall back to weight inflation: the 5 spatial conv kernels from
torchvision's ImageNet-pretrained 2D AlexNet are inflated to 3D by repeating
them along the temporal axis and re-scaling by 1/T (the I3D "inflate" trick).
FC layers are NOT transferred because the spatial resolution differs (224×224
for 2D ImageNet vs. 112×112 here), so they are initialised with Kaiming normal.

References
----------
Krizhevsky et al. (2012) "ImageNet Classification with Deep CNNs"
Carreira & Zisserman (2017) "Quo Vadis, Action Recognition?" (inflation trick)
'''

import os
import warnings
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core architecture
# ---------------------------------------------------------------------------

class AlexNet3D(nn.Module):
    """
    3D AlexNet for video clip classification.

    Expected input shape: (B, 3, T, H, W)
        Typical usage in this codebase: T=16, H=W=112

    Feature tensor shape after global average pooling: (B, 256)
    This is what `test_svm=True` returns, consistent with other models here.
    """

    def __init__(self, num_classes: int = 600,
                 sample_size: int = 112,
                 sample_duration: int = 16):
        super(AlexNet3D, self).__init__()
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Convolutional backbone
        # Spatial strides mirror the 2D AlexNet; temporal strides are kept
        # conservative (1 or 2) so a 16-frame clip is not collapsed too fast.
        #
        # Traced output sizes for (B, 3, 16, 112, 112):
        #   after conv1+pool1  → (B, 64,  16, 13, 13)
        #   after conv2+pool2  → (B, 192,  8,  6,  6)
        #   after conv3        → (B, 384,  8,  6,  6)
        #   after conv4        → (B, 256,  8,  6,  6)
        #   after conv5+pool3  → (B, 256,  4,  2,  2)
        #   after avgpool      → (B, 256,  1,  1,  1) → flatten → (B, 256)
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1 — conv1
            nn.Conv3d(3, 64,
                      kernel_size=(3, 11, 11),
                      stride=(1, 4, 4),
                      padding=(1, 2, 2),
                      bias=False),                  # 0
            nn.BatchNorm3d(64),                     # 1
            nn.ReLU(inplace=True),                  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2)),         # 3

            # Block 2 — conv2
            nn.Conv3d(64, 192,
                      kernel_size=(3, 5, 5),
                      stride=1,
                      padding=(1, 2, 2),
                      bias=False),                  # 4
            nn.BatchNorm3d(192),                    # 5
            nn.ReLU(inplace=True),                  # 6
            nn.MaxPool3d(kernel_size=(2, 3, 3),
                         stride=(2, 2, 2)),         # 7

            # Block 3 — conv3 (no pool)
            nn.Conv3d(192, 384,
                      kernel_size=(3, 3, 3),
                      stride=1,
                      padding=1,
                      bias=False),                  # 8
            nn.BatchNorm3d(384),                    # 9
            nn.ReLU(inplace=True),                  # 10

            # Block 4 — conv4 (no pool)
            nn.Conv3d(384, 256,
                      kernel_size=(3, 3, 3),
                      stride=1,
                      padding=1,
                      bias=False),                  # 11
            nn.BatchNorm3d(256),                    # 12
            nn.ReLU(inplace=True),                  # 13

            # Block 5 — conv5 + pool3
            nn.Conv3d(256, 256,
                      kernel_size=(3, 3, 3),
                      stride=1,
                      padding=1,
                      bias=False),                  # 14
            nn.BatchNorm3d(256),                    # 15
            nn.ReLU(inplace=True),                  # 16
            nn.MaxPool3d(kernel_size=(2, 3, 3),
                         stride=(2, 2, 2)),         # 17
        )

        # Collapse any remaining spatial / temporal dims → (B, 256)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classifier head  (256 → 4096 → 4096 → num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),                      # 0
            nn.Linear(256, 4096),                   # 1
            nn.ReLU(inplace=True),                  # 2
            nn.Dropout(p=0.5),                      # 3
            nn.Linear(4096, 4096),                  # 4
            nn.ReLU(inplace=True),                  # 5
            nn.Linear(4096, num_classes),           # 6
        )

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, test_svm: bool = False):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)          # (B, 256)

        if test_svm:
            # Return the 256-dim pooled feature (before any FC), consistent
            # with how mobilenet_v1 / shufflenet_v1 expose SVM features.
            return x

        return self.classifier(x)          # (B, num_classes)

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inflate_conv_weight(weight_2d: torch.Tensor, t_dim: int) -> torch.Tensor:
    """
    Inflate a 2D conv weight (O, I, H, W) → 3D (O, I, T, H, W).

    The weight is repeated T times along the new temporal axis and rescaled
    by 1/T so that the net activation magnitude is unchanged at initialisation
    (the trick introduced in the I3D paper, Carreira & Zisserman 2017).
    """
    # unsqueeze at dim-2: (O, I, H, W) → (O, I, 1, H, W)
    # repeat along dim-2: → (O, I, T, H, W)
    return weight_2d.unsqueeze(2).repeat(1, 1, t_dim, 1, 1) / t_dim


# ---------------------------------------------------------------------------
# Factory / loading functions  (mirror the interface of the other models)
# ---------------------------------------------------------------------------

def get_model(**kwargs) -> AlexNet3D:
    return AlexNet3D(**kwargs)


def _load_custom_pretrained(model: AlexNet3D,
                            pretrained_model_path: str,
                            n_classes: int) -> AlexNet3D:
    """Load weights from a custom 3D AlexNet checkpoint (your own trained model)."""
    print(f'Loading custom pretrained AlexNet3D from {pretrained_model_path}')
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')

    # Support both raw state-dicts and checkpoint dicts
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Strip 'module.' prefix if the model was saved with DataParallel
    from collections import OrderedDict
    clean = OrderedDict()
    for k, v in state_dict.items():
        clean[k[7:] if k.startswith('module.') else k] = v

    model.load_state_dict(clean, strict=False)

    # Replace the final linear layer for the target class count
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, n_classes)
    return model


def _inflate_from_imagenet(model: AlexNet3D, n_classes: int) -> AlexNet3D:
    """
    Inflate the 5 spatial conv kernels from torchvision's 2D ImageNet AlexNet
    into the 3D model.  FC layers are NOT transferred (spatial resolution
    mismatch: 224×224 for 2D vs 112×112 here).

    Mapping  (2D features index → 3D features index):
        features.0  (Conv2d, conv1)  →  features.0  (Conv3d, conv1)
        features.3  (Conv2d, conv2)  →  features.4  (Conv3d, conv2)
        features.6  (Conv2d, conv3)  →  features.8  (Conv3d, conv3)
        features.8  (Conv2d, conv4)  →  features.11 (Conv3d, conv4)
        features.10 (Conv2d, conv5)  →  features.14 (Conv3d, conv5)
    """
    try:
        import torchvision
    except ImportError:
        warnings.warn(
            'torchvision is not installed; cannot inflate 2D ImageNet weights. '
            'AlexNet3D will be trained from scratch with Kaiming initialisation.',
            RuntimeWarning,
        )
        return model

    print('No 3D AlexNet pretrained weights exist in the public domain.')
    print('Inflating 2D ImageNet AlexNet (torchvision) conv weights → 3D '
          '(I3D inflation trick).  FC layers are initialised with Kaiming normal.')

    alexnet_2d = torchvision.models.alexnet(weights='IMAGENET1K_V1')
    state_2d = alexnet_2d.state_dict()
    state_3d = model.state_dict()

    # (2D key, 3D key)
    conv_pairs = [
        ('features.0.weight',  'features.0.weight'),   # conv1 (3, 64, 11, 11)
        ('features.3.weight',  'features.4.weight'),   # conv2 (192, 64, 5, 5)
        ('features.6.weight',  'features.8.weight'),   # conv3 (384, 192, 3, 3)
        ('features.8.weight',  'features.11.weight'),  # conv4 (256, 384, 3, 3)
        ('features.10.weight', 'features.14.weight'),  # conv5 (256, 256, 3, 3)
    ]

    for key_2d, key_3d in conv_pairs:
        w2d = state_2d[key_2d]                     # (O, I, H, W)
        t_dim = state_3d[key_3d].shape[2]          # temporal kernel size
        state_3d[key_3d] = _inflate_conv_weight(w2d, t_dim)

    model.load_state_dict(state_3d)

    # Replace the final classifier layer for the target class count
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, n_classes)
    return model


# ------------------------------------------------------------------
# Public loading functions  (same signature as all other models here)
# ------------------------------------------------------------------

def pretrained_alexnet_3d(snippet_duration: int,
                          sample_size: int,
                          n_classes: int,
                          pretrained_model_path) -> AlexNet3D:
    """
    Return a 3D AlexNet ready for fine-tuning.

    Weight loading priority
    -----------------------
    1. If `pretrained_model_path` is a path to an existing file → load that
       checkpoint directly (custom trained 3D AlexNet).
    2. Otherwise → inflate from torchvision's 2D ImageNet AlexNet.
       This is the standard fallback because **no public 3D AlexNet checkpoint
       exists** for Kinetics, Sports-1M, or any comparable video dataset.

    The SVM feature dimension exposed by `forward(x, test_svm=True)` is 256
    (the 256-channel global average-pooled representation after the backbone).
    """
    model = get_model(num_classes=n_classes,
                      sample_size=sample_size,
                      sample_duration=snippet_duration)
    model = model.cuda()

    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        model = _load_custom_pretrained(model, pretrained_model_path, n_classes)
    else:
        if pretrained_model_path:
            warnings.warn(
                f'pretrained_model_path "{pretrained_model_path}" not found. '
                f'Falling back to 2D ImageNet inflation.',
                RuntimeWarning,
            )
        model = _inflate_from_imagenet(model, n_classes)

    model.classifier = model.classifier.cuda()
    return model


def pretrained_alexnet_3d_1(snippet_duration: int,
                             sample_size: int,
                             n_classes: int,
                             pretrained_model_path) -> AlexNet3D:
    """
    Same as `pretrained_alexnet_3d` but replaces the final layer with a
    two-layer projection head  (→ projection_dim → n_classes), mirroring the
    `_1` variants of the other models in this codebase.
    """
    model = pretrained_alexnet_3d(snippet_duration, sample_size,
                                  n_classes, pretrained_model_path)

    in_features = model.classifier[6].in_features
    projection_dim = 2304

    model.classifier[6] = nn.Sequential(
        nn.Linear(in_features, projection_dim),
        nn.Linear(projection_dim, n_classes),
    )
    model.classifier = model.classifier.cuda()
    return model


def get_fine_tuning_parameters(model: AlexNet3D, ft_portion: str):
    if ft_portion == 'complete':
        return model.parameters()

    elif ft_portion == 'last_layer':
        parameters = []
        for k, v in model.named_parameters():
            if 'classifier' in k:
                parameters.append({'params': v})
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError(
            "Unsupported ft_portion: 'complete' or 'last_layer' expected"
        )
