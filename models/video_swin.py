'''Video Swin Transformer in PyTorch.

Wraps torchvision's implementation of "Video Swin Transformer" (Liu et al.,
2021, https://arxiv.org/abs/2106.13230), following the official repo at
https://github.com/SwinTransformer/Video-Swin-Transformer.

PRETRAINED WEIGHTS - GOOD NEWS, UNLIKE vit3d.py
------------------------------------------------
Unlike the 3D ViT added earlier, torchvision ships this exact architecture
WITH genuine Kinetics-400-pretrained weights
(`torchvision.models.video.Swin3D_T/S/B_Weights.KINETICS400_V1`), converted
directly from the official repo's released checkpoints (see
model_zoo.yml / README in the official repo: Swin-T gets 78.8% top-1 on
Kinetics-400 with ImageNet-1K-initialised pretraining, Swin-S/B are larger
variants). No ImageNet-inflation workaround is needed here - these are
actual video-trained weights, not an image-model bolted on.

The first time `pretrained_video_swin(...)` is called with no local
`pretrained_model_path`, torchvision downloads the checkpoint automatically
from `download.pytorch.org` and caches it under `~/.cache/torch/hub/checkpoints`
(standard torchvision `weights=` behaviour - same mechanism as
`ViT_B_16_Weights` in vit3d.py, just with real Kinetics-400 weights instead
of ImageNet ones).

INPUT / OUTPUT CONVENTION
--------------------------
Same as every other backbone in this repo: input is (N, C, T, H, W);
`forward(x, test_svm=False)` returns class logits, or pooled pre-classifier
features when `test_svm=True`.

WHY NO POSITIONAL-EMBEDDING INTERPOLATION (unlike vit3d.py)
-------------------------------------------------------------
Video Swin uses *windowed* self-attention with a relative position bias that
depends only on the window size (default 8x7x7 frames/patches), not on the
overall spatial/temporal resolution - so, unlike the plain ViT, it handles
different `sample_size` / `snippet_duration` values natively (via internal
padding in `PatchEmbed3d`) without needing any position-embedding
interpolation trick.
'''
import os
import types
from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models.video import (
    swin3d_t, Swin3D_T_Weights,
    swin3d_s, Swin3D_S_Weights,
    swin3d_b, Swin3D_B_Weights,
)

from torchvision.models.video.swin_transformer import SwinTransformer3d

_VARIANTS = {
    't': (swin3d_t, Swin3D_T_Weights),  # 28M params  - default, matches resnet_18-ish compute budget
    's': (swin3d_s, Swin3D_S_Weights),  # 50M params
    'b': (swin3d_b, Swin3D_B_Weights),  # 88M params
}


def _embed(self, x):
    """Public entry point that stops right after patch embedding + dropout,
    i.e. right before the 4 windowed-attention stages in `self.features`.
    This is the Video-Swin analogue of `ViT3D.embed()` in vit3d.py, and the
    method to call before handing the tensor to
    `torchvision.models._utils.IntermediateLayerGetter(model.CNN.features, {...})`
    for the RSA/CKA co-training loss (see run_neural_model / run_neural_model_dapello).
    x: (N, C, T, H, W) -> (N, T', H', W', C)
    """
    x = self.patch_embed(x)
    x = self.pos_drop(x)
    return x


def _forward_with_test_svm(self, x, test_svm=False):
    x = self.embed(x)
    x = self.features(x)         # 4 windowed-attention stages + 3 patch-merging downsamples
    x = self.norm(x)
    x = x.permute(0, 4, 1, 2, 3)  # (N, C, T', H', W')
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    if test_svm:
        return x
    return self.classifier(x)

SwinTransformer3d._embed = _embed
SwinTransformer3d._forward_with_test_svm = _forward_with_test_svm


def _build(variant, num_classes, pretrained):
    if variant not in _VARIANTS:
        raise ValueError("Unsupported variant '{}': choose from {}".format(variant, list(_VARIANTS.keys())))
    ctor, weights_enum = _VARIANTS[variant]
    weights = weights_enum.KINETICS400_V1 if pretrained else None
    model = ctor(weights=weights)

    # rename .head -> .classifier to match the naming convention used by
    # every other backbone in this repo (mobilenet/shufflenet/squeezenet
    # all expose `.classifier`; only resnet_A uses `.fc`)
    in_features = model.head.in_features
    del model.head
    model.classifier = nn.Linear(in_features, num_classes)

    # bind the test_svm-aware forward and the embed() hook helper
    # model.embed = types.MethodType(_embed, model)
    # model.forward = types.MethodType(_forward_with_test_svm, model)
    SwinTransformer3d.embed = _embed
    SwinTransformer3d.forward = _forward_with_test_svm
    return model


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()
    elif ft_portion == "last_layer":
        parameters = []
        for k, v in model.named_parameters():
            if 'classifier' in k:
                parameters.append({'params': v})
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters
    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(variant='t', num_classes=400, pretrained=True):
    """Returns the model. `variant` in {'t','s','b'} (Swin-Tiny/Small/Base)."""
    return _build(variant, num_classes, pretrained)


def _load_local_checkpoint(model, pretrained_model_path):
    print('Loading pretrained Video Swin {}'.format(pretrained_model_path))
    pretrain = torch.load(pretrained_model_path, map_location='cpu')
    state_dict = pretrain['state_dict'] if 'state_dict' in pretrain else pretrain
    new_state_dict = OrderedDict()
    for name, val in state_dict.items():
        new_name = name[7:] if name.startswith('module.') else name
        new_state_dict[new_name] = val
    model.load_state_dict(new_state_dict, strict=False)
    return model


def pretrained_video_swin(snippet_duration: int,
                           sample_size: int,
                           n_classes,
                           pretrained_model_path=None,
                           variant='t'):
    """Matches the calling convention of the other `pretrained_*` functions
    in this repo (see mobilenet.py / shufflenet.py / squeezenet.py /
    vit3d.py). `snippet_duration`/`sample_size` are accepted for interface
    parity but aren't needed to build the model - Video Swin's windowed
    attention handles arbitrary spatio-temporal resolutions natively.

    If `pretrained_model_path` points at an existing local file, it's loaded
    as a fine-tuning checkpoint (same `module.`-stripping convention as the
    rest of this repo). Otherwise the genuine torchvision Kinetics-400
    weights are used (downloaded automatically on first call).
    """
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model = _build(variant, num_classes=400, pretrained=False)
        model = _load_local_checkpoint(model, pretrained_model_path)
    else:
        print('Loading genuine Kinetics-400 pretrained Video Swin-{} (torchvision)'.format(variant.upper()))
        model = _build(variant, num_classes=400, pretrained=True)

    model = model.cuda()
    # ---------------------------------------------------------------- #
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, n_classes)
    model.classifier = model.classifier.cuda()
    return model


def pretrained_video_swin_1(snippet_duration: int,
                             sample_size: int,
                             n_classes,
                             pretrained_model_path=None,
                             variant='t'):
    """`_1` projection-head variant, matching the pattern used by
    pretrained_mobilenet_v1_1 / pretrained_shufflenet_v1_1 / etc. for the
    co-training / CKA-alignment path."""
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model = _build(variant, num_classes=400, pretrained=False)
        model = _load_local_checkpoint(model, pretrained_model_path)
    else:
        print('Loading genuine Kinetics-400 pretrained Video Swin-{} (torchvision)'.format(variant.upper()))
        model = _build(variant, num_classes=400, pretrained=True)

    model = model.cuda()
    # ---------------------------------------------------------------- #
    in_features = model.classifier.in_features
    projection_dim = 2304
    model.classifier = nn.Sequential(
        nn.Linear(in_features, projection_dim),  # Projection layer
        nn.Linear(projection_dim, n_classes)      # Final classification layer
    )
    model.classifier = model.classifier.cuda()
    return model
