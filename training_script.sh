#!/bin/bash

# activate the virtual environment, if not already activated
source .venv/bin/activate

# python training/train.py --detector_path ./training/config/detector/effort_ce.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_orthogonal.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_hsic.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_hsic_orthogonal.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_orthogonal_weight.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_hsic_orthogonal_weight.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_counterfactual.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_masked_counterfactual.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1" --no-save_ckpt --no-save_feat
# python training/train.py --detector_path ./training/config/detector/effort_ce_hsic_orthogonal_masked_counterfactual.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
python training/train.py --detector_path ./training/config/detector/effort_ce_masked_counterfactual_backbone.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1" --no-save_ckpt --no-save_feat
sudo shutdown now
