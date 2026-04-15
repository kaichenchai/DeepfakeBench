# python training/train.py --detector_path ./training/config/detector/effort_ce.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_orthogonal.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_hsic.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
# python training/train.py --detector_path ./training/config/detector/effort_ce_hsic_orthogonal.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
python training/train.py --detector_path ./training/config/detector/effort_ce_orthogonal_weight.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
python training/train.py --detector_path ./training/config/detector/effort_ce_hsic_orthogonal_weight.yaml  --train_dataset "Celeb-DF-v1" --test_dataset "Celeb-DF-v1"
sudo shutdown now
