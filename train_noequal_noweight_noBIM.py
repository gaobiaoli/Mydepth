import torch

from model.mymodel1_noweight_noequal_noBIM import PriorBIMDA
from train_noequal import main


class NoBIMModel(PriorBIMDA):
    """Keep the baseline architecture while hiding BIM in every forward pass."""

    no_bim = True

    def forward(self, rgb, da3_depth, bim_depth, bim_valid):
        return super().forward(
            rgb,
            da3_depth,
            torch.zeros_like(bim_depth),
            torch.zeros_like(bim_valid),
        )


if __name__ == "__main__":
    main(NoBIMModel)
