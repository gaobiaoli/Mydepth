# DA3Metric-Large pretrained encoder/DPT with global scale and an adapted F36 residual.
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from loss import priorbim_loss

MODEL_ID = "depth-anything/da3metric-large"
MODEL_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
BIM_LOG_MEAN = 0.4236631010536673
BIM_LOG_STD = 0.7573384621476941


def build_bim_condition(da3_depth, bim_depth, bim_valid):
    valid = (bim_valid > 0.5) & (bim_depth > 1e-3) & torch.isfinite(bim_depth)
    if not bool((torch.isfinite(da3_depth) & (da3_depth > 1e-3)).all()):
        raise ValueError("DA3 depth must be positive and finite")

    log_bim = bim_depth.clamp_min(1e-3).log()
    normalized_bim = (log_bim - BIM_LOG_MEAN) / BIM_LOG_STD
    disagreement = ((log_bim - da3_depth.log()) / 1.5).clamp(-1, 1)
    zeros = torch.zeros_like(bim_depth)
    return torch.cat(
        [
            torch.where(valid, normalized_bim, zeros),
            valid.float(),
            torch.where(valid, disagreement, zeros),
        ],
        dim=1,
    )


def build_adapter_condition(da3_depth, bim_depth, bim_valid, log_scale, size):
    valid = (
        (bim_valid > 0.5)
        & (bim_depth > 1e-3)
        & torch.isfinite(bim_depth)
        & (da3_depth > 1e-3)
        & torch.isfinite(da3_depth)
    )
    mask = valid.float()
    disagreement = (
        bim_depth.clamp_min(1e-3).log()
        - da3_depth.clamp_min(1e-3).log()
        - log_scale.detach().float()
    )
    pooled_mask = F.adaptive_avg_pool2d(mask, size)

    def masked_pool(value):
        pooled = F.adaptive_avg_pool2d(value * mask, size)
        return torch.where(
            pooled_mask > 0,
            pooled / pooled_mask.clamp_min(torch.finfo(value.dtype).eps),
            torch.zeros_like(pooled),
        )

    return torch.cat(
        [
            masked_pool((disagreement / 1.5).clamp(-1, 1)),
            masked_pool((disagreement.abs() / 1.5).clamp(0, 1)),
            pooled_mask,
        ],
        dim=1,
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        for layer in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, value):
        return value + self.conv2(self.act(self.conv1(value)))


class DisagreementAdapter(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.input_projection = nn.Conv2d(3, 32, 3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(3)])
        self.output_projection = nn.Conv2d(32, out_channels, 1)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value):
        value = self.act(self.input_projection(value))
        value = self.residual_blocks(value)
        return self.output_projection(value)


class PriorBIMDA(nn.Module):
    """DA3 encoder/DPT features; cached metric DA3 remains the depth anchor."""

    patch_size = 14

    def __init__(self, dav3):
        super().__init__()
        self.dav3 = dav3
        backbone = dav3.backbone.pretrained
        if (
            backbone.alt_start != -1
            or backbone.rope is not None
            or backbone.cat_token
            or backbone.num_register_tokens
        ):
            raise ValueError(
                "This model requires the monocular DA3Metric-Large backbone"
            )

        hidden_dim = backbone.embed_dim
        channels = dav3.head.scratch.layer3_rn.out_channels
        self.gradient_checkpointing = False
        self.bim_condition_embed = nn.Conv2d(3, hidden_dim, 14, stride=14)
        nn.init.zeros_(self.bim_condition_embed.weight)
        nn.init.zeros_(self.bim_condition_embed.bias)

        self.scale_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.scale_head[1].bias)
        nn.init.zeros_(self.scale_head[4].bias)
        nn.init.normal_(self.scale_head[4].weight, std=1e-3)

        self.low2_head = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.kaiming_normal_(self.low2_head[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.low2_head[0].bias)
        nn.init.zeros_(self.low2_head[2].weight)
        nn.init.zeros_(self.low2_head[2].bias)
        self.calibrated_disagreement_adapter = DisagreementAdapter(channels)

        # Only the two coarse DPT branches participate in F36 decoding.
        dav3.head.requires_grad_(False)
        for stage in (2, 3):
            dav3.head.projects[stage].requires_grad_(True)
            dav3.head.resize_layers[stage].requires_grad_(True)
        for name in ("layer3_rn", "layer4_rn", "refinenet3", "refinenet4"):
            getattr(dav3.head.scratch, name).requires_grad_(True)

        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @classmethod
    def from_pretrained(cls, local_files_only=False):
        from depth_anything_3.api import DepthAnything3

        dav3 = DepthAnything3.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
        ).model
        return cls(dav3)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _early_embeddings(self, rgb, condition):
        rgb = (rgb.float().clamp(0, 1) - self.rgb_mean) / self.rgb_std
        height, width = rgb.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by 14")

        backbone = self.dav3.backbone.pretrained
        dtype = backbone.patch_embed.proj.weight.dtype
        rgb_tokens = backbone.patch_embed(rgb.to(dtype=dtype))
        bim_tokens = self.bim_condition_embed(condition).flatten(2).transpose(1, 2)
        tokens = rgb_tokens + bim_tokens.to(dtype=rgb_tokens.dtype)
        cls = backbone.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return tokens + backbone.interpolate_pos_encoding(tokens, height, width)

    def _encode(self, rgb, condition, return_features=True):
        tokens = self._early_embeddings(rgb, condition)
        backbone = self.dav3.backbone.pretrained
        features = []
        for index, block in enumerate(backbone.blocks):
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if return_features and index in self.dav3.backbone.out_layers:
                features.append(backbone.norm(tokens)[:, 1:])
        return backbone.norm(tokens), features

    def _decode_f36(self, features, height, width):
        head = self.dav3.head
        patch_height, patch_width = height // 14, width // 14
        projected = []
        for stage in (2, 3):
            value = head.norm(features[stage])
            value = value.transpose(1, 2).reshape(
                value.shape[0], value.shape[-1], patch_height, patch_width
            )
            value = head.projects[stage](value)
            if head.pos_embed:
                value = head._add_pos_embed(value, width, height)
            value = head.resize_layers[stage](value)
            projected.append(value)

        scratch = head.scratch
        feature36 = scratch.layer3_rn(projected[0])
        feature18 = scratch.layer4_rn(projected[1])
        top_down = scratch.refinenet4(feature18, size=feature36.shape[-2:])
        fusion36 = scratch.refinenet3
        feature36 = top_down + fusion36.resConfUnit1(feature36)
        # Omit the normal F36 -> F72 upsampling, keeping the native F36 grid.
        return fusion36.out_conv(fusion36.resConfUnit2(feature36))

    def predict_log_scale(self, rgb, da3_depth, bim_depth, bim_valid):
        condition = build_bim_condition(da3_depth, bim_depth, bim_valid)
        tokens, _ = self._encode(rgb, condition, return_features=False)
        descriptor = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
        return self.scale_head(descriptor.float()).view(-1, 1, 1, 1)

    def forward(self, rgb, da3_depth, bim_depth, bim_valid):
        condition = build_bim_condition(da3_depth, bim_depth, bim_valid)
        tokens, features = self._encode(rgb, condition)
        descriptor = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
        log_scale = self.scale_head(descriptor.float()).view(-1, 1, 1, 1)

        feature36 = self._decode_f36(features, rgb.shape[-2], rgb.shape[-1])
        adapter_input = build_adapter_condition(
            da3_depth, bim_depth, bim_valid, log_scale, feature36.shape[-2:]
        )
        dtype = self.calibrated_disagreement_adapter.input_projection.weight.dtype
        delta = self.calibrated_disagreement_adapter(adapter_input.to(dtype=dtype))
        log_residual_native = 0.1 * torch.tanh(
            self.low2_head(feature36 + delta.to(dtype=feature36.dtype))
        )
        log_residual = F.interpolate(
            log_residual_native,
            size=da3_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        scaled_depth = da3_depth.float() * log_scale.exp()
        depth = (scaled_depth * log_residual.float().exp()).clamp(1e-3, 128)
        return {
            "depth": depth,
            "scaled_depth": scaled_depth,
            "log_scale": log_scale,
            "log_residual": log_residual,
            "log_residual_native": log_residual_native,
        }

    def compute_loss(self, output, batch, equivariance_error=None):
        return priorbim_loss(output, batch, equivariance_error)

    def parameter_groups(self, factor=1.0):
        return [
            {"params": self.dav3.backbone.parameters(), "lr": 5e-6 * factor},
            {
                "params": [p for p in self.dav3.head.parameters() if p.requires_grad],
                "lr": 5e-5 * factor,
            },
            {"params": self.bim_condition_embed.parameters(), "lr": 5e-5 * factor},
            {"params": self.scale_head.parameters(), "lr": 5e-5 * factor},
            {
                "params": list(self.low2_head.parameters())
                + list(self.calibrated_disagreement_adapter.parameters()),
                "lr": 5e-5 * factor,
            },
        ]

    def load_checkpoint(self, path: str | Path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        state = saved.get("model", saved) if isinstance(saved, Mapping) else saved
        self.load_state_dict(state, strict=True)
