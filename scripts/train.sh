export TZ="Asia/Seoul"
current_date=$(date +"%m%d-%H%M%S")

save_freq=100

CUDA_VISIBLE_DEVICES=$1 \
python train.py \
  --system AlanineDipeptideImplicit \
  --flow_type internal_coords \
  --train_iters 4000 \
  --num_frames 10000000 \
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
  --wandb \
  --run_name $current_date \
  --saved_dir results/ \
  --log_dir results \
  --print_freq 100 \
  --data_save_frequency 10 \
  --ckpt_freq $save_freq \
  --val_freq $save_freq \
  --plot_freq $save_freq