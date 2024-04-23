# MODEL_DIR="results/test"
MODEL_DIR="results/0423-1616"
CKPT_NAME="model_0.ckpt"

CUDA_VISIBLE_DEVICES=$1 python inference.py \
  --model_dir $MODEL_DIR \
  --ckpt $CKPT_NAME \
  --project tps-latent \
  --torch_device 'cuda'\
  --sampling_method 'mcmc' \
  --wandb
  # --md_device 'CUDA' \