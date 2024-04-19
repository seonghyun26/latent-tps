CUDA_VISIBLE_DEVICES=$1 python train.py \
  --flow_type internal_coords \
  --project tps-latent \
  --torch_device 'cuda'\
  --md_device 'CUDA' \
  --wandb