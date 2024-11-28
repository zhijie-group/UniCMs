
#!/bin/bash
# max_train_steps or num_train_epochs
# 设置环境变量
export NCCL_P2P_DISABLE="1"
export CUDA_LAUNCH_BLOCKING=1

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_IB_DISABLE=1 
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_DIR=/liymai24/sjtu/wx/model/showlab/show-o-512x512
MODEL_DIR2=/liymai24/sjtu/wx/model/showlab/show-o-512x512
# MODEL_DIR2=/home/xck/data/ckpt/showo_labelsfinetune_selfr4_lr1e-4/checkpoint-5750/model
# cd /data/xck/mcm_showo
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --main_process_port 29549 --num_processes 2 /home/chenkai/data/mcm_showo/mcm_showo_final_lora.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file /liymai24/sjtu/xck/Show-o/accelerate_configs/8_gpus_deepspeed_zero3.yaml \
  --main_process_port 29537 --num_processes 4  /liymai24/sjtu/xck/mcm_showo_512/mcllm_final512.py \
  --pretrained_teacher_model=$MODEL_DIR \
  --pretrained_student_model=$MODEL_DIR2 \
  --image_dir="/liymai24/sjtu/xck/image/showo_T512_32-4_lr_1e-5_2" \
  --output_dir="/liymai24/sjtu/xck/ckpt/showo_T512_32-4_lr1e-5_2" \
  --num_train_inferences=4 \
  --lr_scheduler="constant" \
  --mixed_precision=fp16 \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=1 \
  --max_train_samples=583747 \
  --max_train_steps=500000 \
  --dataloader_num_workers=1 \
  --train_shards_path_or_url='/liymai24/sjtu/wx/dataset/coco_train_fil.json' \
  --checkpointing_steps=250 \
  --checkpoints_total_limit=10 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --resume_from_checkpoint=latest \
  --report_to=wandb \
  --seed=45369