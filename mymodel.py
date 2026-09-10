from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
MODEL_REVISION = "9560f57a2f07803ba353bb918d6a6e5e005b9277"
BIM_LOG_MEAN = 0.4236631010536673
BIM_LOG_STD = 0.7573384621476941


#计划方案1 强scale 弱refiner, 训练scale后，冻结DINOv2 backbone和neck，训练refiner
#计划2：合成数据集，利用label = wall,floor 等的区域的gt当成bim prior
#计划3 r36处，汇总特征后

def build_bim_condition(da3_depth, bim_depth, bim_valid):
    """Condition used by the zero-initialized DINOv2 patch projection."""
    valid = (bim_valid > 0.5) & (bim_depth > 1e-3) & torch.isfinite(bim_depth)
    if not bool((torch.isfinite(da3_depth) & (da3_depth > 1e-3)).all()):
        raise ValueError("DA3 depth must be positive and finite")

    log_bim = bim_depth.clamp_min(1e-3).log()
    normalized_bim = (log_bim - BIM_LOG_MEAN) / BIM_LOG_STD
    disagreement = ((log_bim - da3_depth.log()) / 1.5).clamp(-1, 1)
    zeros = torch.zeros_like(bim_depth)
    return torch.cat([
        torch.where(valid, normalized_bim, zeros),
        valid.float(),
        torch.where(valid, disagreement, zeros),
    ], dim=1)


def build_adapter_condition(da3_depth, bim_depth, bim_valid, log_scale, size):
    """Post-scale BIM disagreement pooled onto the native F36 grid."""
    valid = (
        (bim_valid > 0.5)
        & (bim_depth > 1e-3)
        & torch.isfinite(bim_depth)
        & (da3_depth > 1e-3)
        & torch.isfinite(da3_depth)
    )
    mask = valid.float()
    z = bim_depth.clamp_min(1e-3).log() - da3_depth.clamp_min(1e-3).log()
    z = z - log_scale.detach().float()

    pooled_mask = F.adaptive_avg_pool2d(mask, size)

    def masked_pool(x):
        value = F.adaptive_avg_pool2d(x * mask, size)
        return torch.where(
            pooled_mask > 0,
            value / pooled_mask.clamp_min(torch.finfo(x.dtype).eps),
            torch.zeros_like(value),
        )

    return torch.cat([
        masked_pool((z / 1.5).clamp(-1, 1)),
        masked_pool((z.abs() / 1.5).clamp(0, 1)),
        pooled_mask,
    ], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        for layer in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


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

    def forward(self, x):
        x = self.act(self.input_projection(x))
        x = self.residual_blocks(x)
        return self.output_projection(x)


class PriorBIMDA(nn.Module):
    """DAv2 metric depth with global scale and an adapted native-F36 residual."""

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

        self.low2_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.kaiming_normal_(self.low2_head[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.low2_head[0].bias)
        nn.init.zeros_(self.low2_head[2].weight)
        nn.init.zeros_(self.low2_head[2].bias)
        self.calibrated_disagreement_adapter = DisagreementAdapter()

        self.dav2.head.requires_grad_(False)
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
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
        if height % 14 or width % 14:
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
        for stage, hidden in zip(backbone.stage_names, output.hidden_states, strict=True):
            if stage in backbone.out_features:
                if backbone.config.apply_layernorm:
                    hidden = backbone.layernorm(hidden)
                features.append(hidden)
        if len(features) != 4:
            raise RuntimeError(f"Expected four DINOv2 feature maps, got {len(features)}")
        return final_tokens, features

    def _decode_f36(self, features, height, width):
        neck = self.dav2.neck
        maps = neck.reassemble_stage(features, height // 14, width // 14)
        projected = [conv(x) for conv, x in zip(neck.convs, maps, strict=True)]

        fusion18 = neck.fusion_stage.layers[0]
        f18 = fusion18.projection(fusion18.residual_layer2(projected[3]))
        top_down = F.interpolate(
            f18, size=projected[2].shape[-2:], mode="bilinear", align_corners=True
        )
        fusion36 = neck.fusion_stage.layers[1]
        f36 = top_down + fusion36.residual_layer1(projected[2])
        return fusion36.projection(fusion36.residual_layer2(f36))

    def predict_log_scale(self, rgb, da3_depth, bim_depth, bim_valid):
        condition = build_bim_condition(da3_depth, bim_depth, bim_valid)
        tokens, _ = self._encode(rgb, condition, return_features=False)
        descriptor = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
        # return self.scale_head(tokens[:, 1:].mean(1).float()).view(-1, 1, 1, 1)
        # return self.scale_head(tokens[:, 0].float()).view(-1, 1, 1, 1)
        return self.scale_head(descriptor.float()).view(-1, 1, 1, 1)

    def forward(self, rgb, da3_depth, bim_depth, bim_valid):
        condition = build_bim_condition(da3_depth, bim_depth, bim_valid)
        tokens, features = self._encode(rgb, condition)
        descriptor = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
        log_scale = self.scale_head(descriptor.float()).view(-1, 1, 1, 1)
        # log_scale = self.scale_head(tokens[:, 1:].mean(1).float()).view(-1, 1, 1, 1)
        # log_scale = self.scale_head(tokens[:, 0].float()).view(-1, 1, 1, 1)

        f36 = self._decode_f36(features, rgb.shape[-2], rgb.shape[-1])
        adapter_input = build_adapter_condition(
            da3_depth, bim_depth, bim_valid, log_scale, f36.shape[-2:]
        )
        adapter_dtype = self.calibrated_disagreement_adapter.input_projection.weight.dtype
        delta = self.calibrated_disagreement_adapter(adapter_input.to(adapter_dtype))
        log_residual_native = 0.1 * torch.tanh(
            self.low2_head(f36 + delta.to(f36.dtype))
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

    def parameter_groups(self,factor=1.0):
        """Learning rates from the best six-epoch training run."""
        return [
            {"params": self.dav2.backbone.parameters(), "lr": 5e-6 * factor},
            {"params": self.dav2.neck.parameters(), "lr": 5e-5 * factor},
            {"params": self.bim_condition_embed.parameters(), "lr": 5e-5 * factor},
            {"params": self.scale_head.parameters(), "lr": 5e-5 * factor},
            {
                "params": list(self.low2_head.parameters())
                + list(self.calibrated_disagreement_adapter.parameters()),
                "lr": 5e-5 * factor,
            },
        ]

    def load_checkpoint(self, path: str | Path):
        """Load either a released state dict or the original experiment checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        # The old implementation stored a frozen, unused r18 head.
        state = {key: value for key, value in state.items() if not key.startswith("low1_head.")}
        self.load_state_dict(state, strict=True)
