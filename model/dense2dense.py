# PriorDA-style dense-to-dense model: DPT F144 predicts a full-resolution log scale map.
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from loss import depth_weights, masked_frame_mean

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
MODEL_REVISION = "9560f57a2f07803ba353bb918d6a6e5e005b9277"
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


class DenseLogScaleHead(nn.Module):
    """Reuse the pretrained DAv2 spatial head shape and zero-init its output."""

    def __init__(self, pretrained_head):
        super().__init__()
        self.conv1 = copy.deepcopy(pretrained_head.conv1)
        self.conv2 = copy.deepcopy(pretrained_head.conv2)
        self.output = nn.Conv2d(self.conv2.out_channels, 1, 1)
        self.act = nn.ReLU()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature144, output_size):
        value = self.conv1(feature144)
        value = F.interpolate(
            value,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        return self.output(self.act(self.conv2(value)))


class PriorBIMDA(nn.Module):
    """Early-fusion DINOv2-DPT predicting one dense log scale map."""

    patch_size = 14

    def __init__(self, dav2):
        super().__init__()
        self.dav2 = dav2
        self.bim_condition_embed = nn.Conv2d(3, 768, 14, stride=14)
        nn.init.zeros_(self.bim_condition_embed.weight)
        nn.init.zeros_(self.bim_condition_embed.bias)
        self.dense_head = DenseLogScaleHead(dav2.head)
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

    def _encode(self, rgb, condition):
        tokens = self._early_embeddings(rgb, condition)
        backbone = self.dav2.backbone
        output = backbone.encoder(
            tokens,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        features = []
        for stage, hidden in zip(
            backbone.stage_names, output.hidden_states, strict=True
        ):
            if stage in backbone.out_features:
                if backbone.config.apply_layernorm:
                    hidden = backbone.layernorm(hidden)
                features.append(hidden)
        if len(features) != 4:
            raise RuntimeError(
                f"Expected four DINOv2 feature maps, got {len(features)}"
            )
        return features

    def _decode_f144(self, features, height, width):
        neck = self.dav2.neck
        maps = neck.reassemble_stage(features, height // 14, width // 14)
        projected = [conv(x) for conv, x in zip(neck.convs, maps, strict=True)]

        feature = None
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
        return feature

    def forward(self, rgb, da3_depth, bim_depth, bim_valid):
        condition = build_bim_condition(da3_depth, bim_depth, bim_valid)
        features = self._encode(rgb, condition)
        feature144 = self._decode_f144(features, rgb.shape[-2], rgb.shape[-1])
        logits = self.dense_head(feature144, da3_depth.shape[-2:])
        log_scale = logits
        depth = (da3_depth.float() * log_scale.float().exp()).clamp(1e-3, 128)
        return {
            "depth": depth,
            "scaled_depth": da3_depth.float(),
            "log_scale": log_scale,
        }

    def compute_loss(self, output, batch):
        prediction = output["depth"].float()
        target = batch["gt_depth"].float()
        valid = (
            (batch["gt_valid"] > 0)
            & (target > 0)
            & (prediction > 0)
            & torch.isfinite(target)
            & torch.isfinite(prediction)
        )
        weights = depth_weights(batch) * valid.float()
        log_error = (
            prediction.clamp_min(1e-6).log() - target.clamp_min(1e-6).log()
        ).abs()
        pixel_loss = (log_error * weights).sum() / weights.sum().clamp_min(1)
        frame_loss = masked_frame_mean(log_error, weights)
        depth_loss = 0.5 * (pixel_loss + frame_loss)

        log_target = (
            target.clamp_min(1e-6).log()
            - batch["da3_depth"].detach().float().clamp_min(1e-6).log()
        )
        dense_loss = masked_frame_mean(
            F.smooth_l1_loss(
                output["log_scale"].float(),
                log_target,
                reduction="none",
                beta=0.02,
            ),
            valid,
        )
        return {
            "total": depth_loss + 0.5 * dense_loss,
            "depth": depth_loss,
            "dense": dense_loss,
        }

    def parameter_groups(self, factor=1.0):
        return [
            {"params": self.dav2.backbone.parameters(), "lr": 5e-6 * factor},
            {"params": self.dav2.neck.parameters(), "lr": 5e-5 * factor},
            {"params": self.bim_condition_embed.parameters(), "lr": 5e-5 * factor},
            {"params": self.dense_head.parameters(), "lr": 5e-5 * factor},
        ]

    def load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, Mapping)
            else checkpoint
        )
        self.load_state_dict(state, strict=True)
