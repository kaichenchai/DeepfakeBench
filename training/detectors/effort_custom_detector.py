import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from loss.hsic_loss import HSICLoss

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='effort_custom')
class Effort_Custom_Detector(nn.Module):
    def __init__(self, config=None):
        super(Effort_Custom_Detector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        if self.config['loss_functions'].get("requires_counterfactual_backbone", False):
            self.counterfactual_model = self.build_counterfactual_model(self.backbone)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.mse_loss_func = nn.MSELoss()

    def build_backbone(self, config):
        # Download model
        # https://huggingface.co/openai/clip-vit-large-patch14
        
        # mean: [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]
        
        # ViT-L/14 224*224
        clip_model = CLIPModel.from_pretrained("./models--openai--clip-vit-large-patch14/")

        # Apply SVD to self_attn layers only
        # ViT-L/14 224*224: 1024-1
        # TODO test different levels of residual rank
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024-1)

        for name, param in clip_model.vision_model.named_parameters():
            print('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in clip_model.vision_model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in clip_model.vision_model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

        return clip_model.vision_model
    
    def build_counterfactual_model(self, backbone):
        # Zero out residual components to get counterfactual backbone
        cf_model = copy.deepcopy(backbone)
        for module in cf_model.modules():
            if isinstance(module, SVDResidualLinear) and module.S_residual is not None:
                module.S_residual = nn.Parameter(
                    torch.zeros_like(module.S_residual), requires_grad=False
                )
        for param in cf_model.parameters():
            param.requires_grad = False
        return cf_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_orthogonal_loss(self, data_dict: dict = None, pred_dict: dict = None) -> dict:
        # Regularization term
        loss = 0.0
        lambda_reg = 0.1
        orthogonal_losses = []
        for module in self.backbone.modules():
            if isinstance(module, SVDResidualLinear):
                # Apply orthogonal constraints to the U_residual and V_residual matrix
                orthogonal_losses.append(module.compute_orthogonal_loss())
        
        if orthogonal_losses:
            reg_term = sum(orthogonal_losses)
            loss += lambda_reg * reg_term
        
        return loss

    def get_weight_loss(self, data_dict: dict = None, pred_dict: dict = None):
        weight_sum_dict = {}
        num_weight_dict = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                if str(weight_curr.size()) not in weight_sum_dict.keys():
                    weight_sum_dict[str(weight_curr.size())] = weight_curr
                    num_weight_dict[str(weight_curr.size())] = 1
                else:
                    weight_sum_dict[str(weight_curr.size())] += weight_curr
                    num_weight_dict[str(weight_curr.size())] += 1
        
        loss = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss += -torch.mean(S_sum)
        loss /= len(weight_sum_dict.keys())
        return loss
    
    def get_hsic_loss(self, data_dict: dict = None, pred_dict: dict = None) -> dict:
        hsic_losses = []
        for module in self.backbone.modules():
            if isinstance(module, SVDResidualLinear):
                hsic_losses.append(module.compute_hsic_loss())
        
        if hsic_losses:
            loss = sum(hsic_losses) / len(hsic_losses)
        else:
            loss = torch.tensor(0.0, device=next(self.parameters()).device)
        return loss
    
    def get_counterfactual_loss(self, data_dict: dict, pred_dict: dict) -> dict:
        with torch.no_grad():
            # In the same way as self.backbone, but use frozen model instead
            cf_features = self.counterfactual_model(data_dict["image"])["pooler_output"]
        cf_pred = self.head(cf_features)  # Share same head
        # Compare the counterfactual prediction with the original prediction
        counterfactual_loss = self.mse_loss_func(cf_pred, pred_dict['cls'])
        
        return counterfactual_loss
    
    def get_masked_counterfactual_loss(self, data_dict: dict, pred_dict: dict) -> dict:
        # Masked counterfactual loss, we only want to penalise model for deviating from base model on real faces, not fake
        # Otherwise encourages residual model weights to be zero, penalises model for learning
        with torch.no_grad():
            cf_features = self.counterfactual_model(data_dict["image"])["pooler_output"]
        cf_pred = self.head(cf_features)  

        # Only penalize the model for deviating from the base model on REAL faces
        mask_real = (data_dict['label'] == 0)
        
        if mask_real.sum() > 0:
            # Pull predictions together ONLY for real images
            counterfactual_loss = self.mse_loss_func(cf_pred[mask_real], pred_dict['cls'][mask_real])
        else:
            counterfactual_loss = torch.tensor(0.0, device=pred_dict['cls'].device)
            
        return counterfactual_loss
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        cross_entropy_loss = self.loss_func(pred, label)
        
        overall_loss = 0
        overall_loss += cross_entropy_loss
        
        dynamic_losses = {
            'hsic_loss': torch.tensor(0.0, device=pred.device).detach(),
            'weight_loss': torch.tensor(0.0, device=pred.device).detach(),
            'orthogonal_loss': torch.tensor(0.0, device=pred.device).detach(),
            'counterfactual_loss': torch.tensor(0.0, device=pred.device).detach(),
            'masked_counterfactual_loss': torch.tensor(0.0, device=pred.device).detach(),
        }
        
        # Only compute all of these other losses when training
        if self.training:
            for loss_name in self.config["loss_functions"]["selected"]:
                # will need to maintain loss function name consistency
                method_name = f"get_{loss_name}_loss"
                
                if hasattr(self, method_name):
                    loss_method = getattr(self, method_name)
                    loss_cfg = self.config["loss_functions"].get(loss_name, {})
                    lambda_val = loss_cfg.get("lambda", 1.0)
                    
                    loss_val = loss_method(data_dict, pred_dict)
                    scaled_loss = lambda_val * loss_val
                    overall_loss += scaled_loss
                    
                    # update dynamic losses dict for logging
                    key = f"{loss_name}_loss"
                    if key in dynamic_losses:
                        dynamic_losses[key] = scaled_loss.detach()

        # masking for real and fake classification loss
        mask_real = label == 0
        mask_fake = label == 1

        if mask_real.sum() > 0:
            loss_real = self.loss_func(pred[mask_real], label[mask_real])
        else:
            loss_real = torch.tensor(0.0, device=pred.device)

        if mask_fake.sum() > 0:
            loss_fake = self.loss_func(pred[mask_fake], label[mask_fake])
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)

        loss_dict = {
            'overall': overall_loss,
            'real_loss': loss_real.detach(),
            'fake_loss': loss_fake.detach(),
            'cross_entropy_loss': cross_entropy_loss.detach(),
            **dynamic_losses,
        }
                
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict


# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of singular values to freeze (main weight rank)

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        
        # For HSIC loss calculation        
        self.hsic_loss_func = HSICLoss()
        self.cached_main_features = None
        self.cached_residual_features = None

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def original_forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)

    def forward(self, x):
        # Compute main features        
        main_features = F.linear(x, self.weight_main, None)
        
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # calculate residual features
            residual_features = F.linear(x, residual_weight, None)
        else:
            # If residual components are not set, use only the main weight
            residual_features = torch.zeros_like(main_features)

        out = main_features + residual_features
        if self.bias is not None:
            out += self.bias
            
        # Detach main features on cache: weight_main is frozen (requires_grad=False)
        # Therefore don't need to cache later when calculating HSIC loss
        self.cached_main_features = main_features.detach()
        self.cached_residual_features = residual_features
                
        # this is for checking the shapes of the features and that output from forward matches original_forward
        # print(f"main_features shape: {main_features.shape}\nresidual_features shape: {residual_features.shape}")
        # original_out = self.original_forward(x)
        # print(f"original_out shape: {original_out.shape}\nout shape: {out.shape}")
        # assert torch.allclose(out, original_out, rtol=1e-3, atol=1e4), "Output from forward does not match original_forward"
        # print(main_features.numel(), residual_features.numel())

        # output is of shape: main_features shape: torch.Size([16, 257, 1024]), residual_features shape: torch.Size([16, 257, 1024])
        # this maps to [batch, sequence_length, feature_dim]
        # sequence length is 257 because of the 256 14x14 patches + 1 cls token in ViT-L/14 with 224*224 input

        return out
    
    def compute_orthogonal_loss(self):
        # According to the properties of orthogonal matrices: A^TA = I
        UUT_residual = self.U_residual @ self.U_residual.t()
        VVT_residual = self.V_residual @ self.V_residual.t()
        
        # Construct an identity matrix
        UUT_residual_identity = torch.eye(UUT_residual.size(0), device=UUT_residual.device)
        VVT_residual_identity = torch.eye(VVT_residual.size(0), device=VVT_residual.device)
        
        # Frobenius norm
        loss = 0.5 * torch.norm(UUT_residual - UUT_residual_identity, p='fro') + 0.5 * torch.norm(VVT_residual - VVT_residual_identity, p='fro')
        
        return loss
    
    def compute_hsic_loss(self):
        if self.cached_main_features is None or self.cached_residual_features is None:
            return torch.tensor(0.0, device=self.weight_main.device)

        assert self.cached_main_features.shape == self.cached_residual_features.shape, f"Main and residual features must have the same shape for HSIC loss computation: {self.cached_main_features.shape} vs {self.cached_residual_features.shape}"

        # Shape of cached features is [batch, sequence_length, feature_dim].
        # Use the CLS token (index 0) and treat the batch as the samples dimension.
        # This gives m = batch_size (e.g. 8) instead of m = 257 (sequence length),
        main_feat = self.cached_main_features[:, 0, :]    # [batch, feature_dim], already detached when cached in forward
        residual_feat = self.cached_residual_features[:, 0, :]     # [batch, feature_dim]

        # Free the full cached tensors now that we have the slices we need.
        self.cached_main_features = None
        self.cached_residual_features = None

        loss = self.hsic_loss_func(main_feat, residual_feat)

        return loss
        

# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            # S_residual is trainable
            new_module.S_residual = nn.Parameter(S_residual.clone())
            # U_residual and V_residual are also trainable
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
        else:
            # If no residual components, set placeholders
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

        return new_module
    else:
        return module