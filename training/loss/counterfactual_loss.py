import torch
import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="counterfactual")
class CounterfactualLoss(AbstractLossClass):
    def __init__(self, counterfactual_model: nn.Module, head: nn.Module):
        super().__init__()
        self.counterfactual_model = counterfactual_model
        self.head = head

    def generate_counterfactual_output(self, input_data):
        with torch.no_grad():
            feat = self.counterfactual_model(input_data['image'])['pooler_output']
            return self.head(feat)

    def forward(self, pred, label, input_data):
        counterfactual_output = self.generate_counterfactual_output(input_data)
        loss = nn.MSELoss()(pred, counterfactual_output)
        return loss