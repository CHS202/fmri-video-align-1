'''3D / Video Vision Transformer (ViT) in PyTorch.

See "An Image is Worth 16x16 Words: Transformers for Image Recognition at
Scale" (Dosovitskiy et al., 2020) and "ViViT: A Video Vision Transformer"
(Arnab et al., 2021) for background.

IMPORTANT - PRETRAINED WEIGHTS
-------------------------------
Unlike the other backbones in this repo (mobilenet, shufflenet, squeezenet,
resnet_A), there is no publicly available 3D/video ViT checkpoint that is
compatible with this codebase's loading convention (a flat state_dict keyed
by this exact module's layer names, loaded with the "strip the first 7
characters of 'module.'" pattern used everywhere else in this repo).

The video-ViT checkpoints that DO exist publicly (VideoMAE, ViViT, TimeSformer,
MViT) are trained/released in entirely different repos (mmaction2, timm,
HuggingFace transformers) with different architectures, different patch/tubelet
sizes, and different state_dict key names. They cannot be `load_state_dict`'d
into a from-scratch module here without a full manual key-remapping exercise,
and even then the architectures are not identical to what's implemented below.

So, exactly like `alexnet3d.py`'s `pretrained_alexnet_3d` already does for
3D AlexNet in this repo, this file follows the same fallback strategy:

    No compatible 3D ViT checkpoint exists -> inflate a public 2D ImageNet-
    pretrained ViT-B/16 (torchvision) into a video ViT, following the
    "central frame initialisation" inflation strategy described in ViViT
    (Arnab et al., 2021, Sec. 3.4):
      - the patch-embedding Conv2d is inflated to a Conv3d tubelet embedding
        by zero-init at all temporal taps except the centre tap, which gets
        the pretrained 2D kernel;
      - all 12 pretrained transformer encoder blocks (attention + MLP + LN)
        are copied directly, since the encoder itself is patch-count/
        sequence-length agnostic;
      - the classification head is replaced with a fresh linear layer;
      - a new learnable temporal positional embedding is added on top of the
        (tiled) pretrained spatial positional embedding, initialised at zero
        so the model starts out equivalent to frame-independent ViT and can
        learn temporal structure during co-training.

If you later obtain a genuinely compatible checkpoint (e.g. you convert a
VideoMAE checkpoint yourself), `pretrained_vit_3d` will use it automatically
when `pretrained_model_path` points at a file that exists on disk and unpacks
into this module's state_dict layout - see the loading branch below.
'''
import os
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.vision_transformer import (
        vit_b_16, ViT_B_16_Weights, EncoderBlock
    )
    _TORCHVISION_VIT_AVAILABLE = True
except Exception:
    _TORCHVISION_VIT_AVAILABLE = False


class TubeletEmbed3D(nn.Module):
    """Video -> patch tokens via a single Conv3d (tubelet embedding).

    Equivalent to ViViT's "tubelet embedding" tokenisation scheme: each
    token covers `frame_patch_size` frames x `patch_size` x `patch_size`
    pixels.
    """
    def __init__(self, in_channels=3, embed_dim=768, patch_size=16, frame_patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(frame_patch_size, patch_size, patch_size),
            stride=(frame_patch_size, patch_size, patch_size),
        )

    def forward(self, x):
        # x: (N, C, T, H, W) -> (N, embed_dim, T', H', W') -> (N, num_tokens, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT3D(nn.Module):
    def __init__(self,
                 num_classes=600,
                 sample_size=112,
                 snippet_duration=16,
                 patch_size=16,
                 frame_patch_size=2,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_dim=3072,
                 dropout=0.0,
                 attention_dropout=0.0):
        super().__init__()
        assert sample_size % patch_size == 0, "sample_size must be divisible by patch_size"
        assert snippet_duration % frame_patch_size == 0, "snippet_duration must be divisible by frame_patch_size"

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size

        self.n_h = sample_size // patch_size
        self.n_w = sample_size // patch_size
        self.n_t = snippet_duration // frame_patch_size
        self.n_spatial_tokens = self.n_h * self.n_w
        self.n_tokens = self.n_t * self.n_spatial_tokens

        self.patch_embed = TubeletEmbed3D(3, embed_dim, patch_size, frame_patch_size)

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # spatial positional embedding (shared across all temporal groups at init)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.n_spatial_tokens + 1, embed_dim))
        # temporal positional embedding, zero-init -> identity at start of training
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.n_t, embed_dim))

        self.dropout = nn.Dropout(dropout)

        layers = OrderedDict()
        for i in range(depth):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads, embed_dim, mlp_dim, dropout, attention_dropout,
            ) if _TORCHVISION_VIT_AVAILABLE else _FallbackEncoderBlock(
                embed_dim, num_heads, mlp_dim, dropout, attention_dropout
            )
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        # temporal_pos_embed intentionally left at zero-init (see module docstring)

    def _build_tokens(self, x):
        # x: (N, 3, T, H, W)
        N = x.shape[0]
        tokens = self.patch_embed(x)  # (N, n_t * n_h * n_w, embed_dim)
        tokens = tokens.view(N, self.n_t, self.n_spatial_tokens, self.embed_dim)
        tokens = tokens + self.spatial_pos_embed[:, 1:, :].unsqueeze(1)
        tokens = tokens + self.temporal_pos_embed.unsqueeze(2)
        tokens = tokens.reshape(N, self.n_t * self.n_spatial_tokens, self.embed_dim)

        cls = self.class_token.expand(N, -1, -1) + self.spatial_pos_embed[:, :1, :]
        tokens = torch.cat([cls, tokens], dim=1)
        return tokens

    def embed(self, x):
        """Public entry point that stops right before the transformer stack:
        tubelet-embeds the video, adds the spatial+temporal positional
        embeddings and the class token, and applies embedding dropout.

        This is the method the RSA/CKA co-training loss (`run_neural_model`,
        `run_neural_model_dapello`, `get_intermediate_outputs_v0`) should call
        to get a tensor it can feed into
        `torchvision.models._utils.IntermediateLayerGetter(model.CNN.layers, {...})`,
        exactly the way it feeds `model.input_process(visual)` output into
        `model.CNN.features` for the CNN backbones. `self.layers` is an
        `nn.Sequential` of `EncoderBlock`s named `encoder_layer_0` ...
        `encoder_layer_{depth-1}`, so it is a drop-in target for
        `IntermediateLayerGetter` just like `model.CNN.features` is for the
        CNNs.
        """
        tokens = self._build_tokens(x)
        tokens = self.dropout(tokens)
        return tokens

    def forward(self, x, test_svm=False):
        tokens = self.embed(x)
        tokens = self.layers(tokens)
        tokens = self.ln(tokens)
        cls_out = tokens[:, 0]
        if test_svm:
            return cls_out
        return self.classifier(cls_out)


class _FallbackEncoderBlock(nn.Module):
    """Used only if torchvision's EncoderBlock can't be imported.
    Standard pre-LN transformer block, so pretrained torchvision weights
    can still be copied into it (same submodule names)."""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, attention_dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.ln_1(x)
        y, _ = self.self_attention(y, y, y, need_weights=False)
        x = x + self.dropout(y)
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


def _inflate_patch_embed(conv2d_weight, frame_patch_size):
    """ViViT 'central frame initialisation' (Arnab et al. 2021, Sec 3.4, Eq 9):
    zero at every temporal tap except the centre one, which takes the
    pretrained 2D kernel. conv2d_weight: (embed_dim, 3, P, P)."""
    embed_dim, in_ch, p, _ = conv2d_weight.shape
    conv3d_weight = torch.zeros(embed_dim, in_ch, frame_patch_size, p, p)
    centre = frame_patch_size // 2
    conv3d_weight[:, :, centre, :, :] = conv2d_weight
    return conv3d_weight


def _load_imagenet_vit_inflated(model: ViT3D):
    """Inflate torchvision's ImageNet-pretrained ViT-B/16 into `model`."""
    if not _TORCHVISION_VIT_AVAILABLE:
        print("WARNING: torchvision ViT not available; ViT3D initialised from scratch (random init).")
        return model

    src = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    with torch.no_grad():
        # 1. patch embedding: Conv2d -> Conv3d, central-frame inflation
        model.patch_embed.proj.weight.copy_(
            _inflate_patch_embed(src.conv_proj.weight, model.frame_patch_size)
        )
        if src.conv_proj.bias is not None and model.patch_embed.proj.bias is not None:
            model.patch_embed.proj.bias.copy_(src.conv_proj.bias)

        # 2. class token
        model.class_token.copy_(src.class_token)

        # 3. spatial positional embedding: interpolate if grid size differs
        src_pos = src.encoder.pos_embedding  # (1, 1+14*14, 768) for 224/16
        n_src_spatial = src_pos.shape[1] - 1
        if n_src_spatial == model.n_spatial_tokens:
            model.spatial_pos_embed.copy_(src_pos)
        else:
            cls_pos = src_pos[:, :1, :]
            grid_pos = src_pos[:, 1:, :]
            side = int(math.sqrt(n_src_spatial))
            grid_pos = grid_pos.reshape(1, side, side, -1).permute(0, 3, 1, 2)
            grid_pos = F.interpolate(grid_pos, size=(model.n_h, model.n_w), mode='bicubic', align_corners=False)
            grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, model.n_spatial_tokens, -1)
            model.spatial_pos_embed.copy_(torch.cat([cls_pos, grid_pos], dim=1))

        # temporal_pos_embed stays at zero-init -> model starts equivalent to
        # frame-independent ViT applied to every frame (see docstring).

        # 4. transformer encoder blocks: copy directly, layer-for-layer
        for i in range(len(model.layers)):
            dst_block = model.layers[i]
            src_block = src.encoder.layers[i]
            dst_block.load_state_dict(src_block.state_dict())

        # 5. final layernorm
        model.ln.load_state_dict(src.encoder.ln.state_dict())

    print("ViT3D initialised via ViViT-style inflation of torchvision ViT-B/16 (ImageNet-1k).")
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


def get_model(**kwargs):
    model = ViT3D(**kwargs)
    return model


def pretrained_vit_3d(snippet_duration: int,
                       sample_size: int,
                       n_classes,
                       pretrained_model_path):
    """Matches the calling convention of the other `pretrained_*` functions
    in this repo (see mobilenet.py / shufflenet.py / squeezenet.py)."""
    model = get_model(num_classes=600, sample_size=sample_size, snippet_duration=snippet_duration)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        # A genuinely compatible checkpoint (e.g. one you've converted yourself)
        print('Loading pretrained 3D ViT {}'.format(pretrained_model_path))
        pretrain = torch.load(pretrained_model_path, map_location='cpu')
        state_dict = pretrain['state_dict'] if 'state_dict' in pretrain else pretrain
        new_state_dict = OrderedDict()
        for name, val in state_dict.items():
            new_name = name[7:] if name.startswith('module.') else name
            new_state_dict[new_name] = val
        model.load_state_dict(new_state_dict, strict=False)
    else:
        # No compatible Kinetics ViT checkpoint exists for this architecture -
        # fall back to ImageNet ViT-B/16 inflation (see module docstring, and
        # the identical fallback already used by alexnet3d.py in this repo).
        model = _load_imagenet_vit_inflated(model)

    model = model.cuda()
    # ---------------------------------------------------------------- #
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    model.classifier = model.classifier.cuda()
    return model


def pretrained_vit_3d_1(snippet_duration: int,
                         sample_size: int,
                         n_classes,
                         pretrained_model_path):
    """Variant with a projection head, matching the `_1` pattern used by
    the other backbones (e.g. pretrained_mobilenet_v1_1) for the co-training
    / CKA-alignment path."""
    model = get_model(num_classes=600, sample_size=sample_size, snippet_duration=snippet_duration)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print('Loading pretrained 3D ViT {}'.format(pretrained_model_path))
        pretrain = torch.load(pretrained_model_path, map_location='cpu')
        state_dict = pretrain['state_dict'] if 'state_dict' in pretrain else pretrain
        new_state_dict = OrderedDict()
        for name, val in state_dict.items():
            new_name = name[7:] if name.startswith('module.') else name
            new_state_dict[new_name] = val
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model = _load_imagenet_vit_inflated(model)

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
