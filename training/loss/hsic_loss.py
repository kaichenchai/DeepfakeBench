import torch
import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="hsic")
class HSICLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        
    def pairwise_distances(self, x):
        #x should be two dimensional
        instances_norm = torch.sum(x**2,-1).reshape((-1,1))
        return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

    def median_bandwidth(self, dists):
        # Median heuristic: sigma = median of upper-triangle pairwise distances.
        # Mask excludes the diagonal and lower triangle
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        upper = dists[mask]
        sigma = upper.median()
        # Guard against degenerate cases (all identical inputs)
        return sigma.clamp(min=1e-3)

    def GaussianKernelMatrix(self, x, sigma=None):
        pairwise_distances_ = self.pairwise_distances(x)
        if sigma is None:
            sigma = self.median_bandwidth(pairwise_distances_)
        return torch.exp(-pairwise_distances_ / sigma)

    def HSIC(self, x, y, s_x=None, s_y=None):
        m,_ = x.shape #batch size
        if m <= 1:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
            
        K = self.GaussianKernelMatrix(x,s_x)
        L = self.GaussianKernelMatrix(y,s_y)
        device = x.device
        dtype = x.dtype
        H = torch.eye(m, device=device, dtype=dtype) - (1.0/m) * torch.ones((m, m), device=device, dtype=dtype)
        HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
        return HSIC
        
    def forward(self, frozen_features, residual_features):
        # Ensure frozen_features and residual_features are 2D - for HSIC calculation with torch.mm
        if frozen_features.dim() == 1:
            frozen_features = frozen_features.unsqueeze(1)
        if residual_features.dim() == 1:
            residual_features = residual_features.unsqueeze(1)

        hsic_loss = self.HSIC(frozen_features, residual_features)
        return hsic_loss
    
    
if __name__ == "__main__":
    # Example usage
    hsic_loss_func = HSICLoss()
    x = torch.randn(257, 1024)  # 257 samples, 1024 features each
    y = torch.randn(257, 1024)  # 257 samples, 1024 features each
    loss = hsic_loss_func(x, y)
    print(f"HSIC Loss: {loss.item()}")