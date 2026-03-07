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

    def GaussianKernelMatrix(self, x, sigma=1):
        pairwise_distances_ = self.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ /sigma)

    def HSIC(self, x, y, s_x=1, s_y=1):
        m,_ = x.shape #batch size
        K = self.GaussianKernelMatrix(x,s_x)
        L = self.GaussianKernelMatrix(y,s_y)
        device = x.device
        dtype = x.dtype
        H = torch.eye(m, device=device, dtype=dtype) - (1.0/m) * torch.ones((m, m), device=device, dtype=dtype)
        HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
        return HSIC
        
    def forward(self, pred, label):
        hsic_loss = self.HSIC(pred, label)
        return hsic_loss
    
    
if __name__ == "__main__":
    pred = torch.randn(10, 512)
    label = torch.randn(10, 512)
    hsic_loss_func = HSICLoss()
    loss = hsic_loss_func(pred, label)
    print(loss)