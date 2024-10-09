#!/usr/bin/env sh

train_data_root='configs/data/image2image_s1.yaml'

model=PixWizard_2B_GQA_patch2
batch_size=1024
lr=1e-4
precision=bf16
image_size=512-768
vae=sdxl
seed=3400
sdxl_vae_path=/path/to/stabilityai/sdxl-vae
tokenizer_path=/path/to/gemma-2b
clip_tokenizer_path=models/clip/clip-vit-large-patch14-336
init_from=/path/to/your/checkpoint
load_type=Lumina     # PixWizard  /  Lumina

exp_name=s1_${model}_bs${batch_size}_lr${lr}_${precision}_${image_size}px_vae${vae}_seed${seed}
mkdir -p results/"$exp_name"

srun -p Gvlab --gres=gpu:8 --quotatype=spot -N 1 --ntasks-per-node 8 --cpus-per-task 12 python -W ignore -u train.py \
    --master_port 18188 \
    --model ${model} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --micro_batch_size 2 \
    --global_batch_size ${batch_size} \
    --lr ${lr} \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 500 --log_every 2 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --image_size ${image_size} \
    --load_type $load_type \
    --init_from $init_from \
    --global_seed ${seed} \
    --vae ${vae} \
    --sdxl_vae_path ${sdxl_vae_path} \
    --tokenizer_path ${tokenizer_path} \
    --clip_tokenizer_path ${clip_tokenizer_path} \
    --num_workers 16 \
    --cache_data_on_disk \
    2>&1 | tee -a results/"$exp_name"/output.log

# --resume $resume_from \
