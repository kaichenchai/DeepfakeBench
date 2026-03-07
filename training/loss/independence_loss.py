import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC
from loss.hsic_loss import HSICLoss
from loss.counterfactual_loss import CounterfactualLoss

@LOSSFUNC.register_module(module_name="independence")
class IndependenceLoss(AbstractLossClass):
    def __init__(self, ce_weight=1.0, hsic_weight=1.0, cf_weight=1.0):
        super().__init__()
        self.cs_weight = ce_weight
        self.hsic_weight = hsic_weight
        self.cf_weight = cf_weight

        self.cross_entropy = nn.CrossEntropyLoss()
        self.hsic = HSICLoss()
        self.counterfactual = CounterfactualLoss()
        
    def forward(self, pred, label, input_data):
        cross_entropy_loss = self.cross_entropy(pred, label)
        hsic_loss = self.hsic(pred, label)
        counterfactual_loss = self.counterfactual(pred, label, input_data)

        total_loss = (self.cs_weight * cross_entropy_loss +
                      self.hsic_weight * hsic_loss +
                      self.cf_weight * counterfactual_loss)

        return total_loss