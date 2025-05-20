
#!/bin/bash
export NCCL_P2P_DISABLE="1"
export CUDA_LAUNCH_BLOCKING=1

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_IB_DISABLE=1 

MODEL_DIR=/showlab/show-o
MODEL_DIR2=/showlab/show-o

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/8_gpus_deepspeed_zero3.yaml \
  --main_process_port 29537 --num_processes 4  UniCMs/train_script/train512.py \
  --pretrained_teacher_model=$MODEL_DIR \
  --pretrained_student_model=$MODEL_DIR2 \
  --image_dir="image/UniCMs" \
  --output_dir="ckpt/UniCMs" \
  --num_train_inferences=4 \
  --lr_scheduler="constant" \
  --mixed_precision=fp16 \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=1 \
  --max_train_samples=583747 \
  --max_train_steps=500000 \
  --dataloader_num_workers=1 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=10 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --resume_from_checkpoint=latest \
  --report_to=wandb \
  --seed=45369