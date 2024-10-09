#!/usr/bin/env sh

ckpt=/path/to/ckpt
name=PixWizard_2B
res=768

json_path=infer_examples/test.json

sdxl_vae_path=/path/to/sdxl-vae
tokenizer_path=/path/to/gemma-2b
clip_tokenizer_path=models/clip/clip-vit-large-patch14-336

CUDA_VISIBLE_DEVICES=5, python -u sample_pixwizard.py ODE \
    --ckpt ${ckpt} \
    --image_save_path generated_images/${name}/${res} \
    --sampling-method euler \
    --num_sampling_steps 60 \
    --seed 3400 \
    --resolution ${res} \
    --json_path ${json_path} \
    --sdxl_vae_path ${sdxl_vae_path} \
    --tokenizer_path ${tokenizer_path} \
    --clip_tokenizer_path ${clip_tokenizer_path} \
    --time_shifting_factor 1 \
    --text_cfg_scale 4.0 \
    --image_cfg_scale 1.0 \
    --ema