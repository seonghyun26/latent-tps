CUDA_VISIBLE_DEVICES=$1 python train.py \
  --system AlanineDipeptideImplicit \
  --flow_type internal_coords \
  --num_frames 100000 \
  --hidden_dim 256 \
  --update_layers 12 \
  --batch_size 1024 \
  --weight_decay 1.e-5 \
  --lr 5.e-4 \
  --lr_schedule cosine \
  --warmup_dur 1000 \
  --kl_loss_weight 1 \
  --rkl_loss_weight 0 \
  --grad_clip 1000 \
  --torch_device cuda \
  --md_device OpenCL \
  --project tps-latent \
  --run_name ALDP_RKL0_KL1_h256_u12_warmup_lrcosine_rerun \
  --wandb

