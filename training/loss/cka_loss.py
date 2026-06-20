"""
Hilbert-Schmidt Independence Criterion (HSIC) loss, normalized as Centered
Kernel Alignment (CKA) for stable, interpretable training.

Reference implementations:
- Kornblith et al. "Similarity of neural network representations revisited"
  (ICML 2019) — https://arxiv.org/abs/1905.00414
- Nguyen et al. "Do wide and deep networks learn the same things?" (2020)
  — https://arxiv.org/abs/2010.15327
- ckatorch: https://github.com/RistoAle97/centered-kernel-alignment
"""

import torch
import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="cka")
class CKALoss(AbstractLossClass):
    """
    CKA loss function based on the Hilbert-Schmidt Independence Criterion (HSIC) with linear kernel.
    This implementation uses an unbiased estimator independent of batch size, and a minibatch version of CKA
    """

    def __init__(self):
        """
        Args:
            threshold: fraction of the median squared distance to use as RBF
                       kernel bandwidth.  Default 1.0 (standard choice).
            unbiased:  if True (default), use the unbiased HSIC₁ estimator
                       whose expected value does *not* depend on batch size.
                       If False, use the biased HSIC₀ (simpler, lower
                       variance but batch-size-dependent).
        """
        super().__init__()

    @staticmethod
    def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
        """Compute the batched version of the Hilbert-Schmidt Independence Criterion on Gram matrices.

        This version is based on
        https://github.com/numpee/CKA.pytorch/blob/07874ec7e219ad29a29ee8d5ebdada0e1156cf9f/cka.py#L107.

        Args:
            gram_x (torch.Tensor): batch of Gram matrices of shape (bsz, n, n).
            gram_y (torch.Tensor): batch of Gram matrices of shape (bsz, n, n).

        Returns:
            torch.Tensor: a tensor with the unbiased Hilbert-Schmidt Independence Criterion values.

        Raises:
            ValueError: if ``gram_x`` and ``gram_y`` do not have the same shape or if they do not have exactly three
                dimensions.
        """
        if len(gram_x.size()) != 3 or gram_x.size() != gram_y.size():
            raise ValueError("Invalid size for one of the two input tensors.")

        n = gram_x.shape[-1]
        gram_x = gram_x.clone()
        gram_y = gram_y.clone()

        # Fill the diagonal of each matrix with 0
        gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
        gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)

        # Compute the product between k (i.e.: gram_x) and l (i.e.: gram_y)
        kl = torch.bmm(gram_x, gram_y)

        # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
        trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

        # Compute the middle term
        middle_term = gram_x.sum((-1, -2), keepdim=True) * gram_y.sum((-1, -2), keepdim=True)
        middle_term /= (n - 1) * (n - 2)

        # Compute the right term
        right_term = kl.sum((-1, -2), keepdim=True)
        right_term *= 2 / (n - 2)

        # Put all together to compute the main term
        main_term = trace_kl + middle_term - right_term

        # Compute the hsic values
        out = main_term / (n**2 - 3 * n)
        return out.squeeze(-1).squeeze(-1)

    def CKA(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the minibatch version of CKA from Nguyen et al. (https://arxiv.org/abs/2010.15327).

        This computation is performed with linear kernel and by calculating HSIC_1.

        Args:
            x (torch.Tensor): tensor of shape (bsz, n, j).
            y (torch.Tensor): tensor of shape (bsz, n, k).

        Returns:
            torch.Tensor: a float tensor in [0, 1] that is the CKA value between the two given tensors.
        """
        x = x.type(torch.float64) if x.dtype != torch.float64 else x
        y = y.type(torch.float64) if y.dtype != torch.float64 else y

        # Build the Gram matrices by applying the linear kernel
        gram_x = torch.bmm(x, x.transpose(1, 2))
        gram_y = torch.bmm(y, y.transpose(1, 2))

        # Compute the HSIC values for the entire batches
        hsic1_xy = self.hsic1(gram_x, gram_y)
        hsic1_xx = self.hsic1(gram_x, gram_x)
        hsic1_yy = self.hsic1(gram_y, gram_y)

        # Unbiased HSIC₁ can produce negative self-HSIC values due to
        # estimator variance.  Clamp the denominator sums to ≥ 0 so that
        # sqrt() doesn't return NaN, and add a tiny epsilon to prevent
        # infinite gradient (1/√0) in the backward pass.
        sum_xx = hsic1_xx.sum().clamp(min=0.0)
        sum_yy = hsic1_yy.sum().clamp(min=0.0)
        denom = (sum_xx * sum_yy + 1e-12).sqrt()
        if denom < 1e-12:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Compute the CKA value and clamp to the theoretical [0, 1] range
        cka = hsic1_xy.sum() / denom
        return cka.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:  (m, d) or (bsz, m, d) — features.
            y:  (m, d) or (bsz, m, d) — features.

        Returns:
            Scalar CKA ∈ [0, 1].  0 → independent, 1 → identical up to scale.
        """
        # Promote 2D → 3D: (n, d) → (1, n, d)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        return self.CKA(x, y)


# ------------------------------------------------------------------
# Quick sanity checks
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    fn = CKALoss()

    # 1. Independent features → CKA near 0 (lower is better)
    x = torch.randn(500, 1024)
    y = torch.randn(500, 1024)
    print(f"  CKA independent (m=500):  {fn(x, y).item():.6f}")

    # 2. Identical features → CKA ≈ 1
    print(f"  CKA identical   (m=500):  {fn(x, x).item():.6f}")

    # 3. Small batch — should NOT differ much from large batch if unbiased
    fn2 = CKALoss()
    x_s = torch.randn(8, 16, 1024)
    y_s = torch.randn(8, 16, 1024)
    print(f"  CKA independent (m=16):   {fn2(x_s, y_s).item():.6f}")
    print(f"  CKA identical   (m=16):   {fn2(x_s, x_s).item():.6f}")

    # 4. All-tokens scenario (m = 4112 ≈ 16 × 257)
    fn3 = CKALoss()
    x_big = torch.randn(8, 4112, 1024)
    y_big = torch.randn(8, 4112, 1024)
    print(f"  CKA independent (m=4112): {fn3(x_big, y_big).item():.6f}")
    print(f"  CKA identical   (m=4112): {fn3(x_big, x_big).item():.6f}")

    # 5. Half-independent features — first half of dims shared, second half random
    x = torch.randn(500, 1024)
    y_half = x.clone()
    y_half[:, 512:] = torch.randn(500, 512)  # overwrite second half with noise
    print(f"  CKA half-shared (m=500):  {fn(x, y_half).item():.6f}  (expect ~0.5)")

    # 6. Noisy copy — y = x + σ·noise, varying SNR
    for sigma in (0.01, 0.1, 1.0, 10.0):
        y_noisy = x + sigma * torch.randn_like(x)
        print(f"  CKA noisy σ={sigma:<5} (m=500):  {fn(x, y_noisy).item():.6f}")