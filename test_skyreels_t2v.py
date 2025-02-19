import os
import argparse
import torch
from transformers import CLIPTextModel, CLIPTokenizerFast, LlamaModel, LlamaTokenizerFast
# from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers import HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from pipelines.pipeline_skyreels_t2v import HunyuanVideoPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description = "HunyuanVideo lora test script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="./models",
        help="Path to pretrained model base directory",
        )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./",
        help = "Output directory for results"
        )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for inference"
        )
    parser.add_argument(
        "--width",
        type = int,
        default = 512,
        help = "Width for inference"
        )
    parser.add_argument(
        "--height",
        type = int,
        default = 512,
        help = "Width for inference"
        )
    parser.add_argument(
        "--num_frames",
        type = int,
        default = 33,
        help = "Number of frames per video, must be divisible by 4+1"
        )
    parser.add_argument(
        "--inference_steps",
        type = int,
        default = 20,
        help = "Number of steps for inference",
        )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A person typing on a laptop keyboard",
        help="Prompt for inference",
        )
    
    args = parser.parse_args()
    return args

@torch.inference_mode()
def main(args):
    
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model,
        subfolder = "transformer-skyreels-t2v",
        torch_dtype = torch.bfloat16,
    )
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.pretrained_model,
        transformer=transformer,
        torch_dtype=torch.float16,
    )
    
    pipe.vae.enable_tiling(
        tile_sample_min_height = 256,
        tile_sample_min_width = 256,
        tile_sample_min_num_frames = 64,
        tile_sample_stride_height = 192,
        tile_sample_stride_width = 192,
        tile_sample_stride_num_frames = 16,
    )
    
    pipe.enable_sequential_cpu_offload()
    # pipe.scheduler.set_shift(17.0)

    output = pipe(
        prompt = args.prompt,
        guidance_scale = 1.0,
        cfg_scale = 6.0,
        cfg_steps = 5,
        height = args.height,
        width = args.width,
        num_frames = args.num_frames,
        num_inference_steps = args.inference_steps,
        generator = torch.Generator(device="cpu").manual_seed(args.seed),
    ).frames[0]
    
    export_to_video(
        output,
        os.path.join(args.output_dir, "output_skyreels_t2v.mp4"),
        fps=15,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)