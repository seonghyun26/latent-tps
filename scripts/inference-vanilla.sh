ckpt_file=""

CUDA_VISIBLE_DEVICES=$1 python inference.py \
  --run_name mcmc_prob_langevin_40_noise0.05_seed0 \
  --sampling_method mcmc \
  --model_dir ./results/best \
  --ckpt $ckpt_file.ckpt \
  --path_density langevin \
  --noise_scale 0.05 \
  --num_steps 40 \
  --langevin_timestep 40 \
  --num_paths 100 \
  --seed 0