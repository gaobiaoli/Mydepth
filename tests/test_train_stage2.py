import json
import sys

import pytest
import torch

import train_stage2


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.batches = 0

    def load_checkpoint(self, path):
        pass

    def parameter_groups(self, factor):
        return [{"params": self.parameters(), "lr": 5e-5 * factor}]

    def forward(self, rgb, da3, bim, valid):
        self.batches += 1
        return {"log_scale": self.weight.expand(1, 1, 1, 1)}

    def predict_log_scale(self, rgb, da3, bim, valid):
        return self.weight.expand(1, 1, 1, 1)

    def compute_loss(self, output, batch, equivariance):
        return {"total": (self.weight - 1).square()}


@pytest.mark.parametrize(
    "options, expected_iters, expected_batches, skip_first",
    [
        (["--iter", "2", "--epochs", "99"], [2], 4, False),
        (["--iter", "4", "--epochs", "0"], [3, 4], 7, False),
        (["--epochs", "2"], [3, 6], 10, False),
        (["--iter", "4"], [2, 4], 9, True),
    ],
)
def test_training_limit(
    tmp_path, monkeypatch, options, expected_iters, expected_batches, skip_first
):
    checkpoint = tmp_path / "pretrained" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    output = tmp_path / "run"
    model = TinyModel()
    batch = {
        key: torch.ones(1, 1, 2, 2)
        for key in ("rgb", "da3_depth", "bim_depth", "bim_valid")
    }
    monkeypatch.setattr(
        train_stage2, "build_loaders", lambda *args: ([batch] * 5, [], [])
    )
    monkeypatch.setattr(
        train_stage2.PriorBIMDA, "from_pretrained", lambda **kwargs: model
    )
    monkeypatch.setattr(train_stage2, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_stage2,
        "evaluate",
        lambda *args: {"final": {"abs_rel": 0.1, "rmse": 0.2, "delta1": 0.9}},
    )
    tested = []
    monkeypatch.setattr(
        train_stage2, "evaluate_test", lambda *args: tested.append(True)
    )
    if skip_first:
        real_scaler = torch.amp.GradScaler

        class SkipFirstScaler:
            def __init__(self, *args, **kwargs):
                self.scaler = real_scaler(*args, **kwargs)
                self.calls = 0
                self.scale_value = 2.0

            def __getattr__(self, name):
                return getattr(self.scaler, name)

            def get_scale(self):
                return self.scale_value

            def step(self, optimizer):
                self.calls += 1
                if self.calls > 1:
                    optimizer.step()

            def update(self):
                if self.calls == 1:
                    self.scale_value = 1.0

        monkeypatch.setattr(torch.amp, "GradScaler", SkipFirstScaler)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_stage2.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--accumulation",
            "2",
            "--no-zero-shot",
            *options,
        ],
    )
    train_stage2.main()

    history = json.loads((output / "history.json").read_text())
    saved = torch.load(output / "latest.pt", weights_only=False)
    assert [row["iter"] for row in history] == expected_iters
    assert model.batches == expected_batches
    assert saved["scheduler"]["T_max"] == expected_iters[-1]
    assert saved["scheduler"]["last_epoch"] == expected_iters[-1]
    assert tested == [True]
