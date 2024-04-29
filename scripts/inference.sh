MODEL_DIR=0423-1642
CKPT_NAME=model_900

CUDA_VISIBLE_DEVICES=$1 python inference.py \
  --wandb \
  --project tps-latent \
  --model_dir results/$MODEL_DIR \
  --ckpt $CKPT_NAME.ckpt \
  --num_paths 16 \
  --num_steps 5000 \
  --sampling_method mcmc \
  --run_name $MODEL_DIR-inference \
  --md_device OpenCL \
  --torch_device cuda