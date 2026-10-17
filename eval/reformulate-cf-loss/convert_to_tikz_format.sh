source /data/curtin_eecms/curtin_yanda/deepfakes/DeepfakeBench/.venv/bin/activate

python3 eval/reformulate-cf-loss/npy_to_tikz.py \
    --real eval/reformulate-cf-loss/both_hist/only_ce_both/cos_sim_reals.npy \
    --fake eval/reformulate-cf-loss/both_hist/only_ce_both/cos_sim_fakes.npy

python3 eval/reformulate-cf-loss/npy_to_tikz.py \
    --real eval/reformulate-cf-loss/both_hist/original_normalised_mse_10_scale_relu/cos_sim_reals.npy \
    --fake eval/reformulate-cf-loss/both_hist/original_normalised_mse_10_scale_relu/cos_sim_fakes.npy

python3 eval/reformulate-cf-loss/npy_to_tikz.py \
    --real eval/reformulate-cf-loss/both_hist/original_normalised_mse_100_scale_relu/cos_sim_reals.npy \
    --fake eval/reformulate-cf-loss/both_hist/original_normalised_mse_100_scale_relu/cos_sim_fakes.npy

python3 eval/reformulate-cf-loss/npy_to_tikz.py \
    --real eval/reformulate-cf-loss/both_hist/fixed_normalised_mse_relu/cos_sim_reals.npy \
    --fake eval/reformulate-cf-loss/both_hist/fixed_normalised_mse_relu/cos_sim_fakes.npy

python3 eval/reformulate-cf-loss/npy_to_tikz.py \
    --real eval/reformulate-cf-loss/both_hist/real_only_norm_mse/cos_sim_reals.npy \
    --fake eval/reformulate-cf-loss/both_hist/real_only_norm_mse/cos_sim_fakes.npy

python3 eval/reformulate-cf-loss/npy_to_tikz.py \
    --real eval/reformulate-cf-loss/both_hist/fake_only_relu/cos_sim_reals.npy \
    --fake eval/reformulate-cf-loss/both_hist/fake_only_relu/cos_sim_fakes.npy