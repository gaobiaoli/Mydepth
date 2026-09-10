import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = None

    def forward(self, rgb, da3, bim, bim_valid):
        raise NotImplementedError()
    