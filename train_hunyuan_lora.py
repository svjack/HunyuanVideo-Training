import os
import gc
import random
import numpy as np
import argparse
import datetime
from tqdm.auto import tqdm
import decord
from contextlib import contextmanager
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2, InterpolationMode
from safetensors.torch import load_file, save_file

from transformers import CLIPTextModel, CLIPTokenizerFast, LlamaModel, LlamaTokenizerFast
from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


class VideoDataset(Dataset):
    def __init__(self, root_folder, resolution=512, num_frames=17, limit_samples=None):
        self.root_folder = root_folder
        self.resolution = resolution
        self.num_frames = num_frames
        self.videos_folders = os.listdir(self.root_folder)
        
        if limit_samples is not None:
            stride = len(self.videos_folders) // limit_samples
            self.videos_folders = self.videos_folders[::stride]
            self.videos_folders = self.videos_folders[:limit_samples]
        
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                size = self.resolution,
                interpolation = InterpolationMode.BILINEAR,
                antialias = True,
            ),
            v2.CenterCrop(size=self.resolution),
        ])
    
    def __len__(self):
        return len(self.videos_folders)
    
    def __getitem__(self, idx):
        video_id = self.videos_folders[idx]
        video_folder = os.path.join(self.root_folder, video_id)
        video_file = os.path.join(video_folder, video_id) + ".mp4"
        
        vr = decord.VideoReader(video_file)
        stride = min(random.randint(1, 4), len(vr) // self.num_frames)
        
        if stride > 0:
            seg_len = stride * self.num_frames
            start_frame = random.randint(0, len(vr) - seg_len)
            pixels = vr[start_frame : start_frame+seg_len : stride]
        else:
            pixels = vr[:]
        
        while pixels.shape[0] < self.num_frames:
            pixels = torch.cat([pixels, pixels[-1].unsqueeze(0)], dim=0)
        
        pixels = pixels.movedim(3, 1).unsqueeze(0).contiguous() # FHWC -> FCHW -> BFCHW
        pixels = self.transforms(pixels) * 2 - 1
        pixels = torch.clamp(torch.nan_to_num(pixels), min=-1, max=1)
        
        embedding_file = ["caption_original_hyv.safetensors", "caption_florence_hyv.safetensors"][random.randint(0, 1)]
        embedding_dict = load_file(os.path.join(video_folder, embedding_file))
        
        return {"pixels": pixels, "embedding_dict": embedding_dict}


@contextmanager
def temp_rng(new_seed=None):
	"""
    https://github.com/fpgaminer/bigasp-training/blob/main/utils.py#L73
	Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
	If new_seed is not None, the RNG state is set to this value before the context is entered.
	"""

	# Save RNG state
	old_torch_rng_state = torch.get_rng_state()
	old_torch_cuda_rng_state = torch.cuda.get_rng_state()
	old_numpy_rng_state = np.random.get_state()
	old_python_rng_state = random.getstate()

	# Set new seed
	if new_seed is not None:
		torch.manual_seed(new_seed)
		torch.cuda.manual_seed(new_seed)
		np.random.seed(new_seed)
		random.seed(new_seed)

	yield

	# Restore RNG state
	torch.set_rng_state(old_torch_rng_state)
	torch.cuda.set_rng_state(old_torch_cuda_rng_state)
	np.random.set_state(old_numpy_rng_state)
	random.setstate(old_python_rng_state)


@contextmanager
def timer(message=""):
    start_time = perf_counter()
    yield
    end_time = perf_counter()
    print(f"{message} {end_time - start_time:0.2f} seconds")


def make_dir(base, folder):
    new_dir = os.path.join(base, folder)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_type = "model",
        repo_id = "hunyuanvideo-community/HunyuanVideo",
        local_dir = "./models",
        max_workers = 1,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description = "HunyuanVideo training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--download_model",
        action = "store_true",
        help = "auto download hunyuanvideo-community/HunyuanVideo to ./models if it's missing",
        )
    parser.add_argument(
        "--cache_embeddings",
        action = "store_true",
        help = "preprocess dataset to encode captions",
        )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="./models",
        help="Path to pretrained model base directory",
        )
    parser.add_argument(
        "--train_data_dir",
        type = str,
        default = "E:/datasets/pexels-video/train",
        help = "Path to train folder where each subfolder contains one video and captions",
        )
    parser.add_argument(
        "--val_data_dir",
        type = str,
        default = "E:/datasets/pexels-video/test",
        help = "Path to validation folder where each subfolder contains one video and captions",
        )
    parser.add_argument(
        "--val_samples",
        type = int,
        default = 8,
        help = "Number of videos to use for testing"
        )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs",
        help = "Output directory for training results"
        )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training"
        )
    parser.add_argument(
        "--resolution",
        type = int,
        default = 512,
        help = "Base resolution for training/testing"
        )
    parser.add_argument(
        "--num_frames",
        type = int,
        default = 33,
        help = "Number of frames per video, must be divisible by 4+1"
        )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 1e-3,
        help = "Base learning rate",
        )
    parser.add_argument(
        "--lora_rank",
        type = int,
        default = 8,
        help = "The dimension of the LoRA update matrices",
        )
    parser.add_argument(
        "--test_steps",
        type = int,
        default = 50,
        help = "Validate after every n steps",
        )
    parser.add_argument(
        "--checkpointing_steps",
        type = int,
        default = 50,
        help = "Save a checkpoint of the training state every X steps",
        )
    parser.add_argument(
        "--max_train_steps",
        type = int,
        default = 1_000,
        help = "Total number of training steps",
        )
    
    args = parser.parse_args()
    return args


def cache_embeddings(args):
    print("loading CLIP")
    with timer("loaded CLIP in"):
        tokenizer_clip = CLIPTokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer_2")
        text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder_2").to(device="cuda", dtype=torch.bfloat16)
        text_encoder_clip.requires_grad_(False)
    
    print("loading Llama")
    with timer("loaded Llama in"):
        tokenizer_llama = LlamaTokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer")
        text_encoder_llama = LlamaModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device="cuda", dtype=torch.bfloat16)
        text_encoder_llama.requires_grad_(False)
    
    def encode_clip(prompt):
        input_ids = tokenizer_clip(
            prompt,
            padding = "max_length",
            max_length = 77,
            truncation = True,
            return_tensors = "pt",
        ).input_ids.to(text_encoder_clip.device)
        
        prompt_embeds = text_encoder_clip(
            input_ids,
            output_hidden_states = False,
        ).pooler_output
        
        return prompt_embeds
    
    def encode_llama(
        prompt,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length = 256,
        num_hidden_layers_to_skip = 2,
    ):
        prompt = prompt_template["template"].format(prompt)
        crop_start = prompt_template.get("crop_start", None)
        max_sequence_length += crop_start
        
        text_inputs = tokenizer_llama(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=text_encoder_llama.device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=text_encoder_llama.device)
        
        prompt_embeds = text_encoder_llama(
            input_ids = text_input_ids,
            attention_mask = prompt_attention_mask,
            output_hidden_states = True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]
        
        return prompt_embeds, prompt_attention_mask
    
    def preprocess_captions(dataset_path):
        folders = os.listdir(dataset_path)
        for folder in tqdm(folders):
            video_folder = os.path.join(dataset_path, folder)
            
            for caption_file in ["caption_original.txt", "caption_florence.txt"]:
                caption_path = os.path.join(video_folder, caption_file)
                embedding_path = caption_path.replace(".txt", "_hyv.safetensors")
                
                if not os.path.exists(embedding_path):
                    with open(caption_path, "r") as f:
                        caption = f.read()
                    
                    clip_embed = encode_clip(caption)
                    llama_embed, llama_mask = encode_llama(caption)
                    
                    embedding_dict = {"clip_embed": clip_embed, "llama_embed": llama_embed, "llama_mask": llama_mask}
                    save_file(embedding_dict, embedding_path)
    
    print("preprocessing caption embeddings")
    preprocess_captions(args.val_data_dir)
    preprocess_captions(args.train_data_dir)
    
    del tokenizer_clip, text_encoder_clip, tokenizer_llama, text_encoder_llama
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    decord.bridge.set_bridge('torch')
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_output_dir = make_dir(args.output_dir, date_time)
    checkpoint_dir = make_dir(real_output_dir, "checkpoints")
    t_writer = SummaryWriter(log_dir=real_output_dir, flush_secs=60)
    
    def collate_batch(batch):
        pixels = torch.cat([sample["pixels"] for sample in batch], dim=0)
        clip_embed = torch.cat([sample["embedding_dict"]["clip_embed"] for sample in batch], dim=0)
        llama_embed = torch.cat([sample["embedding_dict"]["llama_embed"] for sample in batch], dim=0)
        llama_mask = torch.cat([sample["embedding_dict"]["llama_mask"] for sample in batch], dim=0)
        return pixels, clip_embed, llama_embed, llama_mask
    
    train_dataset = VideoDataset(
        root_folder = args.train_data_dir,
        resolution = args.resolution,
        num_frames = args.num_frames,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    
    val_dataset = VideoDataset(
        root_folder = args.val_data_dir,
        resolution = args.resolution,
        num_frames = args.num_frames,
        limit_samples = args.val_samples,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = False,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    
    # noise_scheduler = FlowMatchEulerDiscreteScheduler()#.from_pretrained(args.pretrained_model, subfolder="scheduler")
    # print(noise_scheduler.sigmas[500])
    # print(noise_scheduler.timesteps[500])
    
    with timer("loaded VAE in"):
        vae = AutoencoderKLHunyuanVideo.from_pretrained(args.pretrained_model, subfolder="vae").to(device="cuda", dtype=torch.float16)
        vae.requires_grad_(False)
        vae.enable_tiling(
            tile_sample_min_height = 256,
            tile_sample_min_width = 256,
            tile_sample_min_num_frames = 64,
            tile_sample_stride_height = 192,
            tile_sample_stride_width = 192,
            tile_sample_stride_num_frames = 16,
        )
    
    with timer("loaded diffusion model in"):
        # quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True,)
        quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        diffusion_model = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model,
            subfolder = "transformer",
            quantization_config = quant_config,
            torch_dtype = torch.bfloat16,
        )
        
        diffusion_model.requires_grad_(False)
        diffusion_model.enable_gradient_checkpointing()
        torch.cuda.empty_cache()
    
    with timer("added LoRA in"):
        lora_params = []
        attn_blocks = ["transformer_blocks", "single_transformer_blocks"]
        attn_keys = ["to_k", "to_q", "to_v", "to_out.0", "proj_mlp"]
        for name, param in diffusion_model.named_parameters():
            name = name.replace(".weight", "").replace(".bias", "")
            for block in attn_blocks:
                if name.startswith(block):
                    for key in attn_keys:
                        if key in name:
                            lora_params.append(name)
        
        lora_config = LoraConfig(
            r = args.lora_rank,
            lora_alpha = 1,
            init_lora_weights = "gaussian",
            target_modules = lora_params,
        )
        diffusion_model.add_adapter(lora_config)
        
        lora_parameters = []
        total_parameters = 0
        for param in diffusion_model.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
                lora_parameters.append(param)
                total_parameters += param.numel()
    print(f"total trainable parameters: {total_parameters:,}")
    
    # Instead of having just *one* optimizer, we will have a ``dict`` of optimizers
    # for every parameter so we could reference them in our hook.
    optimizer_dict = {p: bnb.optim.AdamW8bit([p], lr=args.learning_rate) for p in lora_parameters}
    
    # Define our hook, which will call the optimizer step() and zero_grad()
    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    
    # Register the hook onto every parameter
    for p in lora_parameters:
        p.register_post_accumulate_grad_hook(optimizer_hook)
    
    gc.collect()
    torch.cuda.empty_cache()
    diffusion_model.train()
    
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps))
    while global_step < args.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            pixels, clip_embed, llama_embed, llama_mask = batch
            
            with torch.inference_mode():
                pixels = pixels.movedim(1, 2).to(device=vae.device, dtype=vae.dtype) # BFCHW -> BCFHW
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                # add sd3/flux timestep density sampling
                sigma = torch.rand(latents.shape[0])
                timesteps = torch.round(sigma * 1000).long()
                sigma = sigma[:, None, None, None, None].to(latents)
                noisy_model_input = noise * sigma + latents * (1 - sigma)
                
                guidance_scale = 1.0
                guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=torch.float32, device="cuda") * 1000.0
            
            torch.cuda.empty_cache()
            pred = diffusion_model(
                hidden_states           =  noisy_model_input.to(device="cuda", dtype=torch.bfloat16),
                timestep                =          timesteps.to(device="cuda", dtype=torch.bfloat16),
                encoder_hidden_states   =        llama_embed.to(device="cuda", dtype=torch.bfloat16),
                encoder_attention_mask  =         llama_mask.to(device="cuda", dtype=torch.bfloat16),
                pooled_projections      =         clip_embed.to(device="cuda", dtype=torch.bfloat16),
                guidance                =           guidance.to(device="cuda", dtype=torch.bfloat16),
                return_dict = False,
            )[0]
            loss = F.mse_loss(pred.float(), (noise - latents).to(pred.device).float())
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step)
            progress_bar.set_postfix({'loss': f"{loss.detach().item():0.3f}"})
            
            loss.backward()
            progress_bar.update(1)
            global_step += 1
            torch.cuda.empty_cache()
            
            # if global_step == 1 or global_step % val_steps == 0:
                # with torch.inference_mode(), temp_rng(val_seed):
                    # val_loss = 0.0
                    # for step, batch in enumerate(tqdm(val_dataloader, desc="validation", leave=False)):
                        # val_loss += get_pred(batch, timestep_range=(100, 900))[0].detach().item()
                    # t_writer.add_scalar("loss/validation", val_loss / len(val_dataloader), global_step * batch_size)
            
            if global_step >= args.max_train_steps or global_step % args.checkpointing_steps == 0:
                save_file(
                    get_peft_model_state_dict(diffusion_model),
                    os.path.join(checkpoint_dir, f"hyv-lora-{global_step:08}.safetensors"),
                )
            
            if global_step >= args.max_train_steps:
                break

# train
    # basic t2i and randn noise to start
    # guidance?


if __name__ == "__main__":
    args = parse_args()
    
    if args.download_model:
        download_model()
        exit()
    
    if args.cache_embeddings:
        cache_embeddings(args)
        exit()
    
    main(args)