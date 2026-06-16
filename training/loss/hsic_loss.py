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


@LOSSFUNC.register_module(module_name="hsic")
class HSICLoss(AbstractLossClass):
    """
    CKA-normalized HSIC loss using an RBF kernel with per‑batch bandwidth.

    Fixes applied over the original implementation:
      1. PER‑BATCH BANDWIDTH – sigma is recomputed from the current batch's
         features on every call, so the kernel adapts to the local distribution.
      2. PROPER RBF FORM – uses exp(-||x-y||² / (2·σ²)) with the median
         heuristic applied to squared distances (as in Kornblith et al.).
      3. CKA NORMALISATION – returns CKA(K,L) = HSIC(K,L)/√(HSIC(K,K)·HSIC(L,L))
         which is bounded in [0,1] and scale-invariant.
      4. FP64 INTERNALS – kernel matrices are computed in float64 for numerical
         stability, then cast back to the input dtype for gradient compatibility.
      5. UNBIASED HSIC₁ (default) – uses Nguyen et al.'s unbiased estimator so
         CKA values are comparable across different batch sizes.  Set
         ``unbiased=False`` to use the simpler biased HSIC₀ instead.

    Usage:
        loss_fn = HSICLoss(threshold=1.0, unbiased=True)   # recommended
        loss_fn = HSICLoss(threshold=1.0, unbiased=False)  # legacy / simpler
        cka_value = loss_fn(frozen_features, residual_features)  # ∈ [0, 1]
    """

    def __init__(self, threshold: float = 1.0, unbiased: bool = True):
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
        self.threshold = threshold
        self.unbiased = unbiased

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_sq_distances(x: torch.Tensor) -> torch.Tensor:
        """
        Squared Euclidean pairwise distances:  D_ij = ||x_i - x_j||².

        Uses the expansion  ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2⟨x_i,x_j⟩
        which is more numerically stable than (x_i - x_j)² broadcasting.
        """
        dot = torch.mm(x, x.t())
        sq_norm = torch.diag(dot)
        return -2.0 * dot + sq_norm[:, None] + sq_norm[None, :]

    @staticmethod
    def _median_sq_bandwidth(sq_dists: torch.Tensor) -> torch.Tensor:
        """Median of the upper-triangle squared pairwise distances."""
        m = sq_dists.shape[0]
        if m <= 1:
            return torch.tensor(1e-6, device=sq_dists.device, dtype=sq_dists.dtype)
        idx = torch.triu_indices(m, m, offset=1, device=sq_dists.device)
        upper = sq_dists[idx[0], idx[1]]
        return upper.median().clamp(min=1e-6)

    def _rbf_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        RBF / Gaussian kernel matrix:

            K_ij = exp( -||x_i - x_j||² / (2 · threshold² · σ²) )

        where σ² is the median squared distance computed from **this call's**
        input x (per-batch, no caching).
        """
        sq_dists = self._pairwise_sq_distances(x)
        sigma_sq = self._median_sq_bandwidth(sq_dists)
        # bandwidth = 2 · threshold² · σ²   (Kornblith et al. convention)
        bandwidth = 2.0 * (self.threshold ** 2) * sigma_sq
        return torch.exp(-sq_dists / bandwidth)

    # ------------------------------------------------------------------
    # HSIC / CKA
    # ------------------------------------------------------------------
    @staticmethod
    def _hsic0(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Biased HSIC estimator (HSIC₀):

            HSIC₀(K, L) = tr(K H L H) / (m - 1)²

        where  H = I_m - (1/m)·1_m·1_mᵀ  is the centering matrix.

        Simple and low-variance, but its expected value depends on m.
        """
        m = K.shape[0]
        if m <= 1:
            return torch.tensor(0.0, device=K.device, dtype=K.dtype)
        device, dtype = K.device, K.dtype
        H = torch.eye(m, device=device, dtype=dtype) - (1.0 / m) * torch.ones(
            (m, m), device=device, dtype=dtype
        )
        return torch.trace(K @ H @ L @ H) / ((m - 1) ** 2)

    @staticmethod
    def _hsic1(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Unbiased HSIC estimator (HSIC₁) — Nguyen et al. (2020).

            HSIC₁(K, L) = 1/(n(n-3)) · [ tr(K̃L̃) + (1ᵀK̃1)(1ᵀL̃1)/((n-1)(n-2))
                                          - (2/(n-2))·1ᵀK̃L̃1 ]

        where K̃, L̃ are K, L with their diagonals set to zero.

        The expected value of HSIC₁ does *not* depend on the batch size n,
        making CKA values comparable across different batch sizes.
        """
        n = K.shape[0]
        if n <= 3:
            # The (n-3) denominator makes the estimator undefined for n ≤ 3.
            return torch.tensor(0.0, device=K.device, dtype=K.dtype)

        # Zero the diagonals (clone to avoid mutating the inputs)
        Kt = K.clone()
        Lt = L.clone()
        Kt.diagonal().zero_()
        Lt.diagonal().zero_()

        # K̃ @ L̃
        kl = Kt @ Lt

        # tr(K̃L̃) — trace of the product
        trace_kl = torch.trace(kl)

        # (1ᵀK̃1) · (1ᵀL̃1) — product of sums of all elements
        sum_k = Kt.sum()
        sum_l = Lt.sum()
        middle_term = sum_k * sum_l / ((n - 1) * (n - 2))

        # 1ᵀK̃L̃1 — sum of all elements of the product
        right_term = kl.sum() * (2.0 / (n - 2))

        main_term = trace_kl + middle_term - right_term
        return main_term / (n * (n - 3))

    def _hsic_fn(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Dispatch to the selected HSIC estimator."""
        return self._hsic1(K, L) if self.unbiased else self._hsic0(K, L)

    def HSIC(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute CKA = HSIC(K, L) / √(HSIC(K, K) · HSIC(L, L)).

        The RBF bandwidth σ² is recomputed per call from the current inputs,
        so the kernel adapts to each batch's feature distribution.

        Returns a scalar in [0, 1] cast to the same dtype as ``x``.
        """
        m, _ = x.shape
        if m <= 1:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # ---- float64 for numerical stability (ckatorch convention) ----
        x64, y64 = x.to(torch.float64), y.to(torch.float64)

        K = self._rbf_kernel(x64)
        L = self._rbf_kernel(y64)

        hsic_kl = self._hsic_fn(K, L)
        hsic_kk = self._hsic_fn(K, K)
        hsic_ll = self._hsic_fn(L, L)

        denom = torch.sqrt(hsic_kk * hsic_ll)
        if denom < 1e-12:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        cka = hsic_kl / denom
        return cka.to(x.dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self, frozen_features: torch.Tensor, residual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            frozen_features:  (m, d) — main-path features (should be detached).
            residual_features: (m, d) — residual-path features.

        Returns:
            Scalar CKA ∈ [0, 1].  0 → independent, 1 → identical up to scale.
        """
        if frozen_features.dim() == 1:
            frozen_features = frozen_features.unsqueeze(1)
        if residual_features.dim() == 1:
            residual_features = residual_features.unsqueeze(1)
        return self.HSIC(frozen_features, residual_features)


# ------------------------------------------------------------------
# Quick sanity checks
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    for unbiased in (False, True):
        label = "HSIC₁ (unbiased)" if unbiased else "HSIC₀ (biased)"
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

        fn = HSICLoss(unbiased=unbiased)

        # 1. Independent features → CKA near 0 (lower is better)
        x = torch.randn(500, 1024)
        y = torch.randn(500, 1024)
        print(f"  CKA independent (m=500):  {fn(x, y).item():.6f}")

        # 2. Identical features → CKA ≈ 1
        print(f"  CKA identical   (m=500):  {fn(x, x).item():.6f}")

        # 3. Small batch — should NOT differ much from large batch if unbiased
        fn2 = HSICLoss(unbiased=unbiased)
        x_s = torch.randn(16, 1024)
        y_s = torch.randn(16, 1024)
        print(f"  CKA independent (m=16):   {fn2(x_s, y_s).item():.6f}")
        print(f"  CKA identical   (m=16):   {fn2(x_s, x_s).item():.6f}")

        # 4. All-tokens scenario (m = 4112 ≈ 16 × 257)
        fn3 = HSICLoss(unbiased=unbiased)
        x_big = torch.randn(4112, 1024)
        y_big = torch.randn(4112, 1024)
        print(f"  CKA independent (m=4112): {fn3(x_big, y_big).item():.6f}")
        print(f"  CKA identical   (m=4112): {fn3(x_big, x_big).item():.6f}")

    print()