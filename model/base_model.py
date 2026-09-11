# 模型抽象基类：规定所有模型必须实现 forward 接口。
import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = None

    def forward(self, rgb, da3, bim, bim_valid):
        raise NotImplementedError()
    
