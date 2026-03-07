import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="independence")
class IndependenceLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()