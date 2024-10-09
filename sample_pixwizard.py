import argparse
import json
import math
import os
import random
import socket
import time
import sys
import functools

from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import models
from transport import Sampler, create_transport
from models.clip.modules import FrozenCLIPEmbedder_Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from data import read_general


resolution2scale = {
    256: [
        "256x256", "128x512", "144x432", "176x368", "192x336", "224x288",
        "288x224", "336x192", "368x176", "432x144", "512x128"
    ],
    512: ['512 x 512', '1024 x 256', '1008 x 256', '992 x 256', '976 x 256', '960 x 256', '960 x 272',
          '944 x 272', '928 x 272', '912 x 272', '896 x 272', '896 x 288', '880 x 288',
          '864 x 288', '848 x 288', '848 x 304', '832 x 304', '816 x 304', '816 x 320',
          '800 x 320', '784 x 320', '768 x 320', '768 x 336', '752 x 336', '736 x 336',
          '736 x 352', '720 x 352', '704 x 352', '704 x 368', '688 x 368', '672 x 368',
          '672 x 384', '656 x 384', '640 x 384', '640 x 400', '624 x 400', '624 x 416',
          '608 x 416', '592 x 416', '592 x 432', '576 x 432', '576 x 448', '560 x 448',
          '560 x 464', '544 x 464', '544 x 480', '528 x 480', '528 x 496', '512 x 496',
          '496 x 512', '496 x 528', '480 x 528', '480 x 544', '464 x 544',
          '464 x 560', '448 x 560', '448 x 576', '432 x 576', '432 x 592', '416 x 592',
          '416 x 608', '416 x 624', '400 x 624', '400 x 640', '384 x 640', '384 x 656',
          '384 x 672', '368 x 672', '368 x 688', '368 x 704', '352 x 704', '352 x 720',
          '352 x 736', '336 x 736', '336 x 752', '336 x 768', '320 x 768', '320 x 784',
          '320 x 800', '320 x 816', '304 x 816', '304 x 832', '304 x 848', '288 x 848',
          '288 x 864', '288 x 880', '288 x 896', '272 x 896', '272 x 912', '272 x 928',
          '272 x 944', '272 x 960', '256 x 960', '256 x 976', '256 x 992', '256 x 1008',
          '256 x 1024'], 
    768: [  "1536 x 384", "1520 x 384", "1504 x 384", "1488 x 384", "1472 x 384", "1472 x 400",
            "1456 x 400", "1440 x 400", "1424 x 400", "1408 x 400", "1408 x 416", "1392 x 416",
            "1376 x 416", "1360 x 416", "1360 x 432", "1344 x 432", "1328 x 432", "1312 x 432",
            "1312 x 448", "1296 x 448", "1280 x 448", "1264 x 448", "1264 x 464", "1248 x 464",
            "1232 x 464", "1216 x 464", "1216 x 480", "1200 x 480", "1184 x 480", "1184 x 496",
            "1168 x 496", "1152 x 496", "1152 x 512", "1136 x 512", "1120 x 512", "1104 x 512",
            "1104 x 528", "1088 x 528", "1072 x 528", "1072 x 544", "1056 x 544", "1040 x 544",
            "1040 x 560", "1024 x 560", "1024 x 576", "1008 x 576", "992 x 576", "992 x 592",
            "976 x 592", "960 x 592", "960 x 608", "944 x 608", "944 x 624", "928 x 624",
            "912 x 624", "912 x 640", "896 x 640", "896 x 656", "880 x 656", "864 x 656",
            "864 x 672", "848 x 672", "848 x 688", "832 x 688", "832 x 704", "816 x 704",
            "816 x 720", "800 x 720", "800 x 736", "784 x 736", "784 x 752", "768 x 752",
            "768 x 768", "752 x 768", "752 x 784", "736 x 784", "736 x 800", "720 x 800",
            "720 x 816", "704 x 816", "704 x 832", "688 x 832", "688 x 848", "672 x 848",
            "672 x 864", "656 x 864", "656 x 880", "656 x 896", "640 x 896", "640 x 912",
            "624 x 912", "624 x 928", "624 x 944", "608 x 944", "608 x 960", "592 x 960",
            "592 x 976", "592 x 992", "576 x 992", "576 x 1008", "576 x 1024", "560 x 1024",
            "560 x 1040", "544 x 1040", "544 x 1056", "544 x 1072", "528 x 1072", "528 x 1088",
            "528 x 1104", "512 x 1104", "512 x 1120", "512 x 1136", "512 x 1152", "496 x 1152",
            "496 x 1168", "496 x 1184", "480 x 1184", "480 x 1200", "480 x 1216", "464 x 1216",
            "464 x 1232", "464 x 1248", "464 x 1264", "448 x 1264", "448 x 1280", "448 x 1296",
            "448 x 1312", "432 x 1312", "432 x 1328", "432 x 1344", "432 x 1360", "416 x 1360",
            "416 x 1376", "416 x 1392", "416 x 1408", "400 x 1408", "400 x 1424", "400 x 1440",
            "400 x 1456", "400 x 1472", "384 x 1472", "384 x 1488", "384 x 1504", "384 x 1520",
            "384 x 1536"],
    1024: ['1024x1024', '2048x512', '2032x512', '2016x512', '2000x512', '1984x512', '1984x528', '1968x528',
           '1952x528', '1936x528', '1920x528', '1920x544', '1904x544', '1888x544', '1872x544',
           '1872x560', '1856x560', '1840x560', '1824x560', '1808x560', '1808x576', '1792x576',
           '1776x576', '1760x576', '1760x592', '1744x592', '1728x592', '1712x592', '1712x608',
           '1696x608', '1680x608', '1680x624', '1664x624', '1648x624', '1632x624', '1632x640',
           '1616x640', '1600x640', '1584x640', '1584x656', '1568x656', '1552x656', '1552x672',
           '1536x672', '1520x672', '1520x688', '1504x688', '1488x688', '1488x704', '1472x704',
           '1456x704', '1456x720', '1440x720', '1424x720', '1424x736', '1408x736', '1392x736',
           '1392x752', '1376x752', '1360x752', '1360x768', '1344x768', '1328x768', '1328x784',
           '1312x784', '1296x784', '1296x800', '1280x800', '1280x816', '1264x816', '1248x816',
           '1248x832', '1232x832', '1232x848', '1216x848', '1200x848', '1200x864', '1184x864',
           '1184x880', '1168x880', '1168x896', '1152x896', '1136x896', '1136x912', '1120x912',
           '1120x928', '1104x928', '1104x944', '1088x944', '1088x960', '1072x960', 
           '1072x976','1056x976', '1056x992', '1040x992', '1040x1008', '1024x1008', '1008x1024',
           '1008x1040', '992x1040', '992x1056', '976x1056', '976x1072', '960x1072', '960x1088',
           '944x1088', '944x1104', '928x1104', '928x1120', '912x1120', '912x1136', '896x1136',
           '896x1152', '896x1168', '880x1168', '880x1184', '864x1184', '864x1200', '848x1200',
           '848x1216', '848x1232', '832x1232', '832x1248', '816x1248', '816x1264', '816x1280',
           '800x1280', '800x1296', '784x1296', '784x1312', '784x1328', '768x1328', '768x1344',
           '768x1360', '752x1360', '752x1376', '752x1392', '736x1392', '736x1408', '736x1424',
           '720x1424', '720x1440', '720x1456', '704x1456', '704x1472', '704x1488', '688x1488',
           '688x1504', '688x1520', '672x1520', '672x1536', '672x1552', '656x1552', '656x1568',
           '656x1584', '640x1584', '640x1600', '640x1616', '640x1632', '624x1632', '624x1648',
           '624x1664', '624x1680', '608x1680', '608x1696', '608x1712', '592x1712', '592x1728',
           '592x1744', '592x1760', '576x1760', '576x1776', '576x1792', '576x1808', '560x1808',
           '560x1824', '560x1840', '560x1856', '560x1872', '544x1872', '544x1888', '544x1904',
           '544x1920', '528x1920', '528x1936', '528x1952', '528x1968', '528x1984', '512x1984',
           '512x2000', '512x2016', '512x2032', '512x2048']
}

def find_closest_size(sizes_list, target_aspect_ratio):
    resolutions = {
        512: 512 * 512,
        768: 768 * 768,
        1024: 1024 * 1024
    }
    min_diff = float('inf')
    closest_aspect_ratio = None
    closest_size = None
    closest_resolution = None

    for size in sizes_list:
        width, height = size.split('x')
        width, height = int(width), int(height)
        area = width * height

        aspect_ratio = width / height
        diff = abs(aspect_ratio - target_aspect_ratio)

        if diff < min_diff:
            min_diff = diff
            closest_aspect_ratio = aspect_ratio
            closest_size = (width, height)
            closest_resolution = min(resolutions, key=lambda x: abs(resolutions[x] - area))

    return closest_aspect_ratio, closest_size, closest_resolution


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

        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        prompt_masks = text_inputs.attention_mask.to(text_encoder.device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks, pooled_prompt_embeds


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--sampling-method",
        type=str,
        default="euler",
        help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq",
    )
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(args.num_gpus)
    torch.cuda.set_device(rank)

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    if dist.get_rank() == 0:
        print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    if dist.get_rank() == 0:
        print(f"Creating lm: Gemma-2B")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos=True)
    tokenizer.padding_side = "right"
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_tokenizer_path)

    text_encoder = AutoModel.from_pretrained(tokenizer_path, torch_dtype=dtype, device_map="cuda").eval()
    cap_feat_dim = text_encoder.config.hidden_size
    clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_tokenizer_path, device_map="cuda").eval()
    cap_clip_feat_dim = clip_text_encoder.config.hidden_size

    if dist.get_rank() == 0:
        print(f"Creating vae: {train_args.vae}")
    if train_args.vae != "sdxl":
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{train_args.vae}", torch_dtype=torch.float32).cuda()
    else:
        if args.sdxl_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(
                args.sdxl_vae_path, torch_dtype=torch.float32,
            ).cuda()
        else:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae", torch_dtype=torch.float32,
            ).cuda()

    clip_Image = FrozenCLIPEmbedder_Image().cuda()

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                             inplace=True),
    ])

    if dist.get_rank() == 0:
        print(f"Creating DiT: {train_args.model}")

    model = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
        cap_clip_feat_dim=cap_clip_feat_dim,
    )
    model.eval().to("cuda", dtype=dtype)

    if args.debug == False:
        # assert train_args.model_parallel_size == args.num_gpus
        if args.ema:
            print("Loading ema model.")
        ckpt = torch.load(
            os.path.join(
                args.ckpt,
                f"consolidated{'_ema' if args.ema else ''}." f"{rank:02d}-of-{args.num_gpus:02d}.pth",
            ),
            map_location="cpu",
        )
        model.load_state_dict(ckpt, strict=True)
        ckpt_path = os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}."
            f"{rank:02d}-of-{args.num_gpus:02d}.pth",
        )
        print(f"load ckpt from {ckpt_path}")

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "images"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.loads(f.read())
        collected_id = []
        for i in info:
            collected_id.append(f'{id(i["caption"])}_{i["resolution"]}')
    else:
        info = []
        collected_id = []

    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    total = len(info)

    if "-" in args.resolution:
        image_size_list = args.resolution.split('-')
    else:
        image_size_list = [args.resolution]
    image_size_list = [int(num) for num in image_size_list]
    print("image resolution center: ", image_size_list)

    with torch.autocast("cuda", dtype):
        for idx, item in enumerate(tqdm(data_list)):
            transport = create_transport(
                args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps
            )
            sampler = Sampler(transport)
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse,
                time_shifting_factor=args.time_shifting_factor,
            )

            if int(args.seed) != 0:
                torch.random.manual_seed(int(args.seed))

            cap = item['prompt']
            img_path = item['input_path']
            gt_img_path = item['target_path']
            if 'task' in item.keys():
                task_name = item['task']
            else:
                task_name = "uk"
            if 'dataset' in item.keys():
                dataset_name = item['dataset']
            else:
                dataset_name = "uk"

            if isinstance(cap, list):
                cap = random.choice(cap)
                cap = "Text to image generation: " + cap

            caps_list = [cap]
            n = len(caps_list)

            if args.image_root:
                inp_img = Image.open(read_general(os.path.join(args.image_root, img_path))).convert("RGB")
                gt_img = Image.open(read_general(os.path.join(args.image_root, gt_img_path))).convert("RGB")
            else:
                inp_img = Image.open(read_general(img_path)).convert("RGB")
                gt_img = Image.open(read_general(gt_img_path)).convert("RGB")
                
            w_gt, h_gt = inp_img.size

            all_resolutions = []
            for res in image_size_list:
                all_resolutions += resolution2scale[res]

            closest_aspect_ratio, closest_size, closest_res = find_closest_size(all_resolutions, w_gt/h_gt)
            w, h = closest_size

            sample_id = f'{idx}_{w}x{h}'

            res_cat = int(closest_res)
            do_extrapolation = res_cat > 1024

            latent_w, latent_h = w // 8, h // 8
            z_1 = torch.randn([1, 4, latent_h, latent_w], device="cuda").to(dtype)
            z = z_1.repeat(n * 3, 1, 1, 1)

            inp_img = inp_img.resize((w, h), resample=Image.LANCZOS)
            save_inp_img = inp_img
            inp_img = image_transform(inp_img).to("cuda")
            
            factor = 0.18215 if train_args.vae != "sdxl" else 0.13025
            input_clip_x = clip_Image.encode(inp_img[None])
            input_clip_x_null = torch.zeros_like(input_clip_x, device="cuda").to(dtype)
            input_x = vae.encode(inp_img[None]).latent_dist.sample().mul_(factor)

            input_clip_x = torch.cat((input_clip_x, input_clip_x, input_clip_x_null), dim=0)
            input_x = torch.cat((input_x, input_x, z_1), dim=0)

            with torch.no_grad():
                cap_feats, cap_mask, cap_pooled_feats = encode_prompt(caps_list + [""] + [""], text_encoder, tokenizer, clip_tokenizer, clip_text_encoder, 0.0)

            cap_mask = cap_mask.to(cap_feats.device)
            cap_pooled_feats = cap_pooled_feats.to(cap_feats.device)

            model_kwargs = dict(
                input_x=input_x,
                input_clip_x=input_clip_x,
                cap_feats=cap_feats,
                cap_mask=cap_mask,
                cap_pooled_feats=cap_pooled_feats,
                text_cfg_scale=args.text_cfg_scale,
                image_cfg_scale=args.image_cfg_scale,
            )

            if args.proportional_attn:
                model_kwargs["proportional_attn"] = True
                model_kwargs["base_seqlen"] = (res_cat // 16) ** 2
            else:
                model_kwargs["proportional_attn"] = False
                model_kwargs["base_seqlen"] = None

            if do_extrapolation and args.scaling_method == "Time-aware":
                model_kwargs["scale_factor"] = math.sqrt(w * h / res_cat**2)
                model_kwargs["scale_watershed"] = args.scaling_watershed
            else:
                model_kwargs["scale_factor"] = 1.0
                model_kwargs["scale_watershed"] = 1.0

            samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples = samples[:n]

            samples = vae.decode(samples / factor).sample
            samples = (samples + 1.0) / 2.0
            samples.clamp_(0.0, 1.0)

            # Save samples to disk as individual .png files
            for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                img = to_pil_image(sample.float())
                save_path = f"{args.image_save_path}/images/{task_name}_{dataset_name}_{sample_id}.png"
                img.save(save_path)
                info.append(
                    {
                        "caption": cap,
                        "image_url": f"{args.image_save_path}/images/{task_name}_{dataset_name}_{sample_id}.png",
                        "resolution": f"res: {args.resolution}\ntime_shift: {args.time_shifting_factor}",
                        "sampling_method": args.sampling_method,
                        "num_sampling_steps": args.num_sampling_steps,
                    }
                )
            save_inp_img.save(f"{args.image_save_path}/images/{task_name}_{dataset_name}_{sample_id}_input.png")
            gt_img = gt_img.resize((w, h), resample=Image.LANCZOS)
            gt_img.save(f"{args.image_save_path}/images/{task_name}_{dataset_name}_{sample_id}_gt.png")
            with open(f"{args.image_save_path}/images/{task_name}_{dataset_name}_{sample_id}_prompt.txt", "w") as file:
                file.write(caps_list[0])
            with open(info_path, "w") as f:
                f.write(json.dumps(info))

            total += len(samples)
            dist.barrier()

    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_cfg_scale", type=float, default=4.0)
    parser.add_argument("--image_cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
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
        "--precision",
        type=str,
        choices=["fp32", "tf32", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.add_argument("--no_ema", action="store_false", dest="ema", help="Do not use EMA models.")
    parser.set_defaults(ema=False)
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--json_path", type=str, default='data.json',
    )
    parser.add_argument(
        "--prompt_txt", type=str, default='prompt.txt',
    )
    parser.add_argument(
        "--image_root", type=str, default='',
    )
    parser.add_argument("--resolution", type=str, required=True)
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="Time-aware",
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)
