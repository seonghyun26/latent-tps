MODEL_DIR=0429-212606
CKPT_NAME=model_3600

CUDA_VISIBLE_DEVICES=$1 python inference.py \
  --wandb \
  --project tps-latent \
  --model_dir results/$MODEL_DIR \
  --ckpt $CKPT_NAME.ckpt \
  --start_state_idx 0 \
  --end_state_idx 4 \
  --sampling_method mcmc \
  --path_density langevin \
  --noise_scale 0.05 \
  --num_steps 40 \
  --langevin_timestep 40 \
  --num_paths 100 \
  --seed 1 \
  --run_name $MODEL_DIR-infer \
  --md_device OpenCL \
  --torch_device cuda