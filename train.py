# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for PixWizard using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time

from PIL import Image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ItemProcessor, MyDataset, read_general
from grad_norm import calculate_l2_grad_norm, get_model_parallel_dim_dict, scale_grad
from imgproc import generate_crop_size_list, var_center_crop
import models
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport
from models.clip.modules import FrozenCLIPEmbedder_Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


#############################################################################
#                            Data item Processor                            #
#############################################################################


class I2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform

    def process_item(self, data_item, training_mode=False):
        input_blank_image_path = random.choice([
                "black_image.png",
                "white_image.png"
            ])
        input_image_path = data_item['input_path']
        if "white_image.png" in input_image_path or "black_image.png" in input_image_path:
            input_image_path = input_blank_image_path

        target_image_path = data_item['target_path']
        input_image = Image.open(read_general(input_image_path)).convert("RGB")
        target_image = Image.open(read_general(target_image_path)).convert("RGB")
        text = data_item.get('prompt', "")

        if isinstance(text, list):
            if len(text) == 0:
                text = ""

            t2i_prefix = [
                "Text to image generation: ",
                "Please generate an image with the following caption: ",
                "Drawing: ",
                "Generate a photo: ",
                "Create an image based on this description: ",
                "Produce an image according to the following description: "
            ]
            text = random.choice(t2i_prefix) + text

        if input_image.size != target_image.size:
            input_image = input_image.resize(target_image.size, Image.LANCZOS)


        input_image = self.image_transform(input_image)
        target_image = self.image_transform(target_image)


        return input_image, target_image, text


#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    input_image = [x[0] for x in samples]
    target_image = [x[1] for x in samples]
    caps = [x[2] for x in samples]
    return input_image, target_image, caps


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    pooled_prompt_embeds = prompt_embeds[0]
    # prompt_embeds = prompt_embeds.hidden_states[-2]
    # prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # _, seq_len, _ = prompt_embeds.shape
    # # duplicate text embeddings for each generation per prompt, using mps friendly method
    # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    # prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return pooled_prompt_embeds


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, clip_tokenizer, clip_text_encoder, 
                    proportion_empty_prompts, is_train=True, num_images_per_prompt: int = 1, device=None):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    clip_pooled_prompt_embeds_list = []
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=clip_text_encoder,
        tokenizer=clip_tokenizer,
        prompt=captions,
        device=device if device is not None else clip_text_encoder.device,
        num_images_per_prompt=num_images_per_prompt,
    )
    clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks, pooled_prompt_embeds


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()

    assert args.global_batch_size % dp_world_size == 0, "Batch size must be divisible by data parrallel world size."
    local_batch_size = args.global_batch_size // dp_world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    # print(f"Starting rank={rank}, seed={seed}, "
    #       f"world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                args.results_dir,
                "tensorboard",
                datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname(),
            )
        )
    else:
        logger = create_logger(None)
        tb_logger = None

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    # create tokenizers
    # Load the tokenizers
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = "google/gemma-2b"

    logger.info(f"Setting-up language model: {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "right"
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_tokenizer_path)

    # create text encoders
    # text_encoder
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            tokenizer_path,
            torch_dtype=torch.bfloat16,
        )
        .get_decoder()
        .cuda()
    )
    text_encoder = setup_lm_fsdp_sync(text_encoder)
    print(f"text encoder: {type(text_encoder)}")
    cap_feat_dim = text_encoder.config.hidden_size

    clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_tokenizer_path).to(text_encoder.device)
    print(f"clip text encoder: {type(text_encoder)}")
    cap_clip_feat_dim = clip_text_encoder.config.hidden_size

    if "-" in args.image_size:
        image_size_list = args.image_size.split('-')
    else:
        image_size_list = [args.image_size]
    image_size_list = [int(num) for num in image_size_list]
    for image_size in image_size_list:
        assert image_size % 8 == 0, (
            "Image size must be divisible by 8 (for the VAE encoder)."
        )

    # Create model:
    model = models.__dict__[args.model](
        in_channels=16 if args.vae == "sd3" else 4,
        qk_norm=args.qk_norm,
        cap_feat_dim=cap_feat_dim,
        cap_clip_feat_dim=cap_clip_feat_dim,
    )
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size

    model_parallel_dim_dict = get_model_parallel_dim_dict(model)

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            state_dict = torch.load(os.path.join(
                args.resume,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
            ), map_location="cpu")
            if args.load_type == "Lumina":
                ### x_embedder init expand
                out_c, in_c = state_dict['x_embedder.weight'].shape
                new_weight = torch.zeros(out_c, in_c * 2)
                new_weight.zero_()
                new_weight[:, :in_c].copy_(state_dict['x_embedder.weight'])
                state_dict['x_embedder.weight'] = new_weight
                ### ---------------------- ###
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Resuming ema weights from: {args.resume}")
            state_dict_ema = torch.load(os.path.join(
                args.resume,
                f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
            ), map_location="cpu")
            if args.load_type == "Lumina":
                ### x_embedder init expand
                out_c, in_c = state_dict_ema['x_embedder.weight'].shape
                new_weight = torch.zeros(out_c, in_c * 2)
                new_weight.zero_()
                new_weight[:, :in_c].copy_(state_dict_ema['x_embedder.weight'])
                state_dict_ema['x_embedder.weight'] = new_weight
                ### ---------------------- ###
            model_ema.load_state_dict(state_dict_ema, strict=True)
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(os.path.join(
                args.init_from,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
            ), map_location="cpu")
            if args.load_type == "Lumina":
                ### x_embedder init expand
                out_c, in_c = state_dict['x_embedder.weight'].shape
                new_weight = torch.zeros(out_c, in_c * 2)
                new_weight.zero_()
                new_weight[:, :in_c].copy_(state_dict['x_embedder.weight'])
                state_dict['x_embedder.weight'] = new_weight
                ### ---------------------- ###
            missing_keys, unexpected_keys = \
                model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = \
                model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.transformer_blocks)
        checkpointing_list_ema = list(model_ema.transformer_blocks)
    else:
        checkpointing_list = []
        checkpointing_list_ema = []

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    # checkpointing (part2, after FSDP wrapping)
    if args.checkpointing:
        print("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )

    logger.info(f"model:\n{model}\n")

    # default: 1000 steps, linear noise schedule
    transport = create_transport("Linear", "velocity", None, None, None, snr_type=args.snr_type)  # default: velocity;
    if args.vae == "sd3":
        logger.info("use SD3 VAE")
        vae = AutoencoderKL.from_pretrained("/path/to/sd3/vae").to(device)
    elif args.vae == "sdxl":
        # vae
        logger.info("use SDXL VAE")
        if args.sdxl_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(
                args.sdxl_vae_path
            ).to(device)
        else:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae"
            ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
            if args.local_diffusers_model_root is None else
            os.path.join(args.local_diffusers_model_root,
                         f"stabilityai/sd-vae-ft-{args.vae}")
        ).to(device)

    clip_Image = FrozenCLIPEmbedder_Image().to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    logger.info("Creating data transform...")
    patch_size = 8 * model_patch_size
    logger.info(f"patch size: {patch_size}")
    logger.info(f"training image size center: {image_size_list}")
    crop_size_list_all = []
    for _, image_size in enumerate(image_size_list):
        max_num_patches = round((image_size / patch_size) ** 2)
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size)

        logger.info(f"Limiting number of patches to {image_size}: {max_num_patches}")
        logger.info(f"List of {image_size} crop sizes:")
        for j in range(0, len(crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[j: j + 6]]))
        
        crop_size_list_all += crop_size_list

    image_transform = transforms.Compose(
        [
            transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list_all, random_top_k=1)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = MyDataset(
        args.data_path,
        item_processor=I2IItemProcessor(image_transform),
        cache_on_disk=args.cache_data_on_disk,
    )
    num_samples = args.global_batch_size * args.max_steps
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Total # samples to consume: {num_samples:,} " f"({num_samples / len(dataset):.2f} epochs)")
    sampler = get_train_sampler(
        dataset,
        dp_rank,
        dp_world_size,
        args.global_batch_size,
        args.max_steps,
        resume_step,
        args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataloader_collate_fn,
    )

    # Prepare models for training:
    # important! This enables embedding dropout for classifier-free guidance
    model.train()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_grad_norm = 0
    start_time = time()

    logger.info(f"Training for {args.max_steps:,} steps...")
    vae_scale = {"sdxl": 0.13025, "sd3": 1.5305, "ema": 0.18215, "mse": 0.18215}[args.vae]
    vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0}[args.vae]
    for step, (input_x, target_x, caps) in enumerate(loader, start=resume_step):
        # caps: List[str]
        input_x = [img_input.to(device, non_blocking=True) for img_input in input_x]
        target_x = [img_target.to(device, non_blocking=True) for img_target in target_x]

        with torch.no_grad():
            if step == resume_step:
                logger.warning(f"vae scale: {vae_scale}    vae shift: {vae_shift}")
            # Map input images to latent space + normalize latents:
            input_clip_x = [clip_Image.encode(img_input[None]) for img_input in input_x]
            mask_1 = torch.bernoulli(torch.full((len(input_clip_x),), 1 - args.image_dropout_prob)).bool()
            for i in range(len(input_clip_x)):
                if mask_1[i]:
                    input_clip_x[i] = torch.zeros_like(input_clip_x[i]).to(device)
            input_x = [(vae.encode(img_input[None]).latent_dist.sample()[0] - vae_shift) * vae_scale for img_input in input_x]
            mask_2 = torch.bernoulli(torch.full((len(input_x),), 1 - args.image_dropout_prob)).bool()
            for i in range(len(input_x)):
                if mask_2[i]:
                    input_x[i] = torch.randn_like(input_x[i]).to(device)
            target_x = [(vae.encode(img_target[None]).latent_dist.sample()[0] - vae_shift) * vae_scale for img_target in target_x]
        with torch.no_grad():
            cap_feats, cap_mask, cap_pooled_feats = encode_prompt(caps, text_encoder, tokenizer, clip_tokenizer, clip_text_encoder, args.caption_dropout_prob)

        loss_item = 0.0
        opt.zero_grad()
        for mb_idx in range((local_batch_size - 1) // args.micro_batch_size + 1):
            mb_st = mb_idx * args.micro_batch_size
            mb_ed = min((mb_idx + 1) * args.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            input_x_mb = input_x[mb_st: mb_ed]
            input_clip_x_mb = input_clip_x[mb_st: mb_ed]
            target_x_mb = target_x[mb_st: mb_ed]

            cap_feats_mb = cap_feats[mb_st: mb_ed]
            cap_mask_mb = cap_mask[mb_st: mb_ed]
            cap_pooled_feats_mb = cap_pooled_feats[mb_st: mb_ed]

            model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb, cap_pooled_feats=cap_pooled_feats_mb)
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = transport.training_losses(model, input_x_mb, input_clip_x_mb, target_x_mb, model_kwargs)
            
            loss = loss_dict["loss"].sum() / local_batch_size
            loss_item += loss.item()
            with model.no_sync() if args.data_parallel in ["sdp", "fsdp"] and not last_mb else contextlib.nullcontext():
                loss.backward()

        grad_norm = calculate_l2_grad_norm(model, model_parallel_dim_dict)
        if grad_norm > args.grad_clip:
            scale_grad(model, args.grad_clip / grad_norm)

        if tb_logger is not None:
            tb_logger.add_scalar("train/loss", loss_item, step)
            tb_logger.add_scalar("train/grad_norm", grad_norm, step)
            tb_logger.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

        opt.step()
        update_ema(model_ema, model)

        # Log loss values:
        running_loss += loss_item
        running_grad_norm += grad_norm
        log_steps += 1
        if (step + 1) % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = args.global_batch_size * log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            avg_grad_norm = running_grad_norm / log_steps
            logger.info(
                f"(step={step + 1:07d}) "
                f"Train Loss: {avg_loss:.4f}, "
                f"Train Grad Norm: {avg_grad_norm:.4f}, "
                f"Train Secs/Step: {secs_per_step:.2f}, "
                f"Train Imgs/Sec: {imgs_per_sec:.2f}"
            )
            # Reset monitoring variables:
            running_loss = 0
            running_grad_norm = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="PixWizard_2B_GQA_patch2")
    parser.add_argument("--image_size", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse", "sdxl", "sd3"], default="ema"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--checkpointing", action="store_true", default=False, help="enable gradient checkpointing")
    parser.add_argument(
        "--local_diffusers_model_root",
        type=str,
        help="Specify the root directory if diffusers models are to be loaded "
        "from the local filesystem (instead of being automatically "
        "downloaded from the Internet). Useful in environments without "
        "Internet access.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--load_type", type=str,
        choices=["Lumina", "PixWizard"],
        help="Load the model weights from the Lumina-T2I checkpoint folder or load from the PixWizard."
    )
    parser.add_argument(
        "--sdxl_vae_path", type=str, default=None, help="local path to sdxl vae"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="local path to tokenizer (Gemma-2b)"
    )
    parser.add_argument(
        "--clip_tokenizer_path", type=str, default=None, help="local path to tokenizer (CLIP-vit-large-patch14)"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=2.0,
        help="Clip the L2 norm of the gradients to the given value.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.05,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument(
        "--image_dropout_prob",
        type=float,
        default=0.05,
        help="Randomly change the caption of a sample to a blank image with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    args = parser.parse_args()

    main(args)