"""
Probe wrapper around ``training/detectors/effort_custom_detector.py``.

Rather than maintaining a duplicate copy of the detector (which drifts out of
sync with the newest training implementation), this module subclasses the real
``Effort_Custom_Detector`` and only adds the probe-only methods used by
``probe_cos_sim.py`` and ``fake_loss_only_probe_real_image_similarity.py``:

    compute_cos_sim          -- per-sample cosine similarity between the frozen
                                counterfactual (pristine CLIP) pooler features
                                and the learned detector pooler features.
    compute_mse              -- per-sample mean squared error (unnormalised).
    compute_normalized_mse   -- per-sample MSE normalised by the batch mean
                                squared L2 norm of the cf features.

Because it inherits from the real detector, the probe is automatically in sync
with the newest training code (including the masked-counterfactual-backbone
ablation toggles ``enable_real_constraint`` / ``enable_fake_constraint``) and
loads checkpoints through the same custom ``state_dict`` / ``load_state_dict``.
The SVD residual backbone, the frozen counterfactual backbone and the per-batch
counterfactual-features cache are all constructed/behaved identically to
training.

The class is registered under the distinct module name ``effort_custom_probe``
so it never collides with the original ``effort_custom`` registration.
"""

import os
import sys

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrap. The real detector lives in <repo>/training/detectors/. Make
# the training package importable even when this module is imported without the
# parent probe scripts having already adjusted sys.path.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
for _p in (SCRIPT_DIR, TRAINING_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detectors import DETECTOR
from detectors.effort_custom_detector import Effort_Custom_Detector


@DETECTOR.register_module(module_name='effort_custom_probe')
class Effort_Custom_Detector_Probe(Effort_Custom_Detector):
    def compute_cos_sim(self, data_dict: dict, pred_dict: dict) -> torch.Tensor:
        # Per-sample cosine similarity between the frozen counterfactual
        # backbone's pooler features and the learned detector pooler features.
        #
        # This is EXACTLY the quantity the masked-counterfactual-backbone
        # losses act on (the fake-branch term in the newest implementation is
        # the temperature-scaled softplus of this cos_sim).
        #
        # Returned un-masked so the caller can select real/fake subsets.
        cf_features = self._get_cf_features(data_dict)
        cos_sim = F.cosine_similarity(cf_features, pred_dict['feat'], dim=-1)
        return cos_sim

    def compute_mse(self, data_dict: dict, pred_dict: dict) -> torch.Tensor:
        # Per-sample mean squared error between the frozen counterfactual
        # backbone pooler features and the learned detector pooler features,
        # averaged over the 1024-d feature dimension. This is the
        # (unnormalised) quantity the real branch minimised before the scale
        # normalisation applied in get_masked_counterfactual_backbone_loss.
        cf_features = self._get_cf_features(data_dict)
        mse = F.mse_loss(cf_features, pred_dict['feat'], reduction='none').mean(dim=-1)
        return mse

    def compute_normalized_mse(self, data_dict: dict, pred_dict: dict) -> torch.Tensor:
        # The real-branch loss normalised the MSE by the batch's mean squared
        # L2 norm of the counterfactual features:
        #   scale = (cf_features.norm(p=2, dim=-1) ** 2).mean().detach() + 1e-8
        #   loss += mse / scale
        # Reproduced per-sample so it can be compared directly with the loss.
        cf_features = self._get_cf_features(data_dict)
        mse = F.mse_loss(cf_features, pred_dict['feat'], reduction='none').mean(dim=-1)
        scale = (cf_features.norm(p=2, dim=-1) ** 2).mean().detach() + 1e-8
        return mse / scale

    def compute_normalized_mse_fixed(self, data_dict: dict, pred_dict: dict) -> torch.Tensor:
        # Mirror of the 'normalized_mse_fixed' real-loss branch in
        # get_masked_counterfactual_backbone_loss: per-sample squared error
        # (summed over the 1024-d feature dim, no 1/D factor) scaled by a single
        # batch-level mean squared L2 norm of the counterfactual features:
        #   d2    = ((cf - pred) ** 2).sum(dim=-1)
        #   scale = (cf ** 2).sum(dim=-1).mean().detach() + 1e-8
        #   return d2 / scale
        # Reproduced per-sample so it can be compared directly with the loss.
        cf_features = self._get_cf_features(data_dict)
        d2 = ((cf_features - pred_dict['feat']) ** 2).sum(dim=-1)
        scale = (cf_features ** 2).sum(dim=-1).mean().detach() + 1e-8
        return d2 / scale
