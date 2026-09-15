# Multi-scale adapter model: independently refine log depth at F36, F72, and F144.
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from loss import priorbim_multiscale_loss

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
MODEL_REVISION = "9560f57a2f07803ba353bb918d6a6e5e005b9277"
BIM_LOG_MEAN = 0.4236631010536673
BIM_LOG_STD = 0.7573384621476941


def build_bim_condition(da3_depth, bim_depth, bim_valid):
    """Condition used by the zero-initialized DINOv2 patch projection."""
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


def build_adapter_condition(
    da3_depth,
    bim_depth,
    bim_valid,
    cumulative_log_correction,
    size,
):
    """Pool BIM disagreement after the previous, detached correction."""
    correction = cumulative_log_correction.detach().float()
    valid = (
        (bim_valid > 0.5)
        & (bim_depth > 1e-3)
        & torch.isfinite(bim_depth)
        & (da3_depth > 1e-3)
        & torch.isfinite(da3_depth)
        & torch.isfinite(correction)
    )
    mask = valid.float()
    disagreement = (
        bim_depth.clamp_min(1e-3).log() - da3_depth.clamp_min(1e-3).log() - correction
    )
    pooled_mask = F.adaptive_avg_pool2d(mask, size)

    def masked_pool(value):
        value = F.adaptive_avg_pool2d(value * mask, size)
        return torch.where(
            pooled_mask > 0,
            value / pooled_mask.clamp_min(torch.finfo(value.dtype).eps),
            torch.zeros_like(value),
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
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv2d(3, 32, 3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(3)])
        self.output_projection = nn.Conv2d(32, 128, 1)
        self.act = nn.GELU()

        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value):
        value = self.act(self.input_projection(value))
        value = self.residual_blocks(value)
        return self.output_projection(value)


class ResidualStage(nn.Module):
    """One independent three-ResBlock disagreement adapter and residual head."""

    def __init__(self):
        super().__init__()
        self.adapter = DisagreementAdapter()
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.kaiming_normal_(self.head[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.head[0].bias)
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, feature, condition):
        dtype = self.adapter.input_projection.weight.dtype
        delta = self.adapter(condition.to(dtype=dtype))
        logits = self.head(feature + delta.to(dtype=feature.dtype))
        return 0.25 * torch.tanh(logits)


class PriorBIMDA(nn.Module):
    """Global scale followed by independent F36, F72, and F144 refiners."""

    patch_size = 14

    def __init__(self, dav2):
        super().__init__()
        self.dav2 = dav2

        self.bim_condition_embed = nn.Conv2d(3, 768, 14, stride=14)
        nn.init.zeros_(self.bim_condition_embed.weight)
        nn.init.zeros_(self.bim_condition_embed.bias)

        self.scale_head = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 256),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.scale_head[1].bias)
        nn.init.zeros_(self.scale_head[4].bias)
        nn.init.normal_(self.scale_head[4].weight, std=1e-3)

        self.stage36 = ResidualStage()
        self.stage72 = ResidualStage()
        self.stage144 = ResidualStage()
        self.dav2.head.requires_grad_(False)

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
        from transformers import AutoModelForDepthEstimation

        dav2 = AutoModelForDepthEstimation.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
        )
        return cls(dav2)

    def enable_gradient_checkpointing(self):
        self.dav2.gradient_checkpointing_enable()

    def _early_embeddings(self, rgb, condition):
        rgb = (rgb.float().clamp(0, 1) - self.rgb_mean) / self.rgb_std
        height, width = rgb.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by 14")

        embeddings = self.dav2.backbone.embeddings
        dtype = embeddings.patch_embeddings.projection.weight.dtype
        rgb_tokens = embeddings.patch_embeddings(rgb.to(dtype=dtype))
        bim_tokens = self.bim_condition_embed(condition).flatten(2).transpose(1, 2)
        tokens = rgb_tokens + bim_tokens.to(rgb_tokens.dtype)
        cls = embeddings.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + embeddings.interpolate_pos_encoding(tokens, height, width)
        return embeddings.dropout(tokens)

    def _encode(self, rgb, condition, return_features=True):
        tokens = self._early_embeddings(rgb, condition)
        backbone = self.dav2.backbone
        output = backbone.encoder(
            tokens,
            output_hidden_states=return_features,
            output_attentions=False,
            return_dict=True,
        )
        final_tokens = backbone.layernorm(output.last_hidden_state)
        if not return_features:
            return final_tokens, ()

        features = []
        for stage, hidden in zip(
            backbone.stage_names,
            output.hidden_states,
            strict=True,
        ):
            if stage in backbone.out_features:
                if backbone.config.apply_layernorm:
                    hidden = backbone.layernorm(hidden)
                features.append(hidden)
        if len(features) != 4:
            raise RuntimeError(
                f"Expected four DINOv2 feature maps, got {len(features)}"
            )
        return final_tokens, features

    def _decode_features(self, features, height, width):
        neck = self.dav2.neck
        maps = neck.reassemble_stage(features, height // 14, width // 14)
        projected = [conv(value) for conv, value in zip(neck.convs, maps, strict=True)]

        feature = None
        decoded = []
        for layer, shortcut in zip(
            neck.fusion_stage.layers,
            reversed(projected),
            strict=True,
        ):
            if feature is None:
                feature = layer.residual_layer2(shortcut)
            else:
                feature = F.interpolate(
                    feature,
                    size=shortcut.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                feature = feature + layer.residual_layer1(shortcut)
                feature = layer.residual_layer2(feature)
            feature = layer.projection(feature)
            decoded.append(feature)

        _, feature36, feature72, feature144 = decoded
        return feature36, feature72, feature144

    @staticmethod
    def _resize(value, size):
        return F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

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

        feature36, feature72, feature144 = self._decode_features(
            features,
            rgb.shape[-2],
            rgb.shape[-1],
        )
        output_size = da3_depth.shape[-2:]

        condition36 = build_adapter_condition(
            da3_depth,
            bim_depth,
            bim_valid,
            log_scale,
            feature36.shape[-2:],
        )
        residual36 = self.stage36(feature36, condition36)
        residual36_full = self._resize(residual36, output_size)

        correction36 = log_scale + residual36_full
        condition72 = build_adapter_condition(
            da3_depth,
            bim_depth,
            bim_valid,
            correction36,
            feature72.shape[-2:],
        )
        residual72 = self.stage72(feature72, condition72)
        residual72_full = self._resize(residual72, output_size)

        correction72 = correction36 + residual72_full
        condition144 = build_adapter_condition(
            da3_depth,
            bim_depth,
            bim_valid,
            correction72,
            feature144.shape[-2:],
        )
        residual144 = self.stage144(feature144, condition144)
        residual144_full = self._resize(residual144, output_size)

        log_residual = residual36_full + residual72_full + residual144_full
        log_residual_native = (
            self._resize(residual36, feature144.shape[-2:])
            + self._resize(residual72, feature144.shape[-2:])
            + residual144
        )
        scaled_depth = da3_depth.float() * log_scale.exp()
        depth = (scaled_depth * log_residual.float().exp()).clamp(1e-3, 128)
        return {
            "depth": depth,
            "scaled_depth": scaled_depth,
            "log_scale": log_scale,
            "log_residual": log_residual,
            "log_residual_native": log_residual_native,
            "log_residual_r36": residual36,
            "log_residual_r72": residual72,
            "log_residual_r144": residual144,
        }

    def compute_loss(self, output, batch, equivariance_error=None):
        return priorbim_multiscale_loss(output, batch, equivariance_error)

    def parameter_groups(self, factor=1.0):
        return [
            {"params": self.dav2.backbone.parameters(), "lr": 5e-6 * factor},
            {"params": self.dav2.neck.parameters(), "lr": 5e-5 * factor},
            {"params": self.bim_condition_embed.parameters(), "lr": 5e-5 * factor},
            {"params": self.scale_head.parameters(), "lr": 5e-5 * factor},
            {
                "params": list(self.stage36.parameters())
                + list(self.stage72.parameters())
                + list(self.stage144.parameters()),
                "lr": 5e-5 * factor,
            },
        ]

    def load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, Mapping)
            else checkpoint
        )
        self.load_state_dict(state, strict=True)
