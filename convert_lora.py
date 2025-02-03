import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_diffusers_to_hunyuan_video_lora(diffusers_state_dict):
    converted_state_dict = {k: diffusers_state_dict.pop(k) for k in list(diffusers_state_dict.keys())}

    TRANSFORMER_KEYS_RENAME_DICT = {
        "img_in": "x_embedder",
        "time_in.mlp.0": "time_text_embed.timestep_embedder.linear_1",
        "time_in.mlp.2": "time_text_embed.timestep_embedder.linear_2",
        "guidance_in.mlp.0": "time_text_embed.guidance_embedder.linear_1",
        "guidance_in.mlp.2": "time_text_embed.guidance_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        ".double_blocks": ".transformer_blocks",
        ".single_blocks": ".single_transformer_blocks",
        "img_attn_q_norm": "attn.norm_q",
        "img_attn_k_norm": "attn.norm_k",
        "img_attn_proj": "attn.to_out.0",
        "txt_attn_q_norm": "attn.norm_added_q",
        "txt_attn_k_norm": "attn.norm_added_k",
        "txt_attn_proj": "attn.to_add_out",
        "img_mod.linear": "norm1.linear",
        "img_norm1": "norm1.norm",
        "img_norm2": "norm2",
        "txt_mlp": "ff_context",
        "img_mlp": "ff",
        "txt_mod.linear": "norm1_context.linear",
        "txt_norm1": "norm1.norm",
        "txt_norm2": "norm2_context",
        "modulation.linear": "norm.linear",
        "pre_norm": "norm.norm",
        "final_layer.norm_final": "norm_out.norm",
        "final_layer.linear": "proj_out",
    #   "linear2": "proj_out",
        "fc1": "net.0.proj",
        "fc2": "net.2",
        "input_embedder": "proj_in",
        # txt_in
        "individual_token_refiner.blocks": "token_refiner.refiner_blocks",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
    #   "t_embedder.mlp.0": "time_text_embed.timestep_embedder.linear_1",
    #   "t_embedder.mlp.2": "time_text_embed.timestep_embedder.linear_2",
        "c_embedder": "time_text_embed.text_embedder",
        "txt_in": "context_embedder",
    #   "mlp": "ff",
    }

    TRANSFORMER_KEYS_RENAME_DICT_REVERSE = {v: k for k, v in TRANSFORMER_KEYS_RENAME_DICT.items()}

    for key in list(converted_state_dict.keys()):
        if "norm_out.linear" in key:
            weight = converted_state_dict.pop(key)
            scale, shift = weight.chunk(2, dim=0)
            new_weight = torch.cat([shift, scale], dim=0)
            converted_state_dict[key] = new_weight

        if "to_q" in key:
            if "single_transformer_blocks" in key:
                to_q = converted_state_dict.pop(key)
                to_k = converted_state_dict.pop(key.replace("to_q", "to_k"))
                to_v = converted_state_dict.pop(key.replace("to_q", "to_v"))
                to_out = converted_state_dict.pop(key.replace("attn.to_q", "proj_mlp"))
                rename_attn_key = "linear1"
                if "lora_A" in key:
                    converted_state_dict[key.replace("attn.to_q", rename_attn_key)] = to_q
                else:
                    qkv_mlp = torch.cat([to_q, to_k, to_v, to_out], dim=0)
                    converted_state_dict[key.replace("attn.to_q", rename_attn_key)] = qkv_mlp
            else:
                to_q = converted_state_dict.pop(key)
                to_k = converted_state_dict.pop(key.replace("to_q", "to_k"))
                to_v = converted_state_dict.pop(key.replace("to_q", "to_v"))
                if "token_refiner" in key:
                    rename_attn_key = "self_attn_qkv"
                    if "lora_A" in key:
                        converted_state_dict[key.replace("attn.to_q", rename_attn_key)] = to_q
                    else:
                        qkv = torch.cat([to_q, to_k, to_v], dim=0)
                        converted_state_dict[key.replace("attn.to_q", rename_attn_key)] = qkv
                else:
                    rename_attn_key = "img_attn_qkv"
                    if "lora_A" in key:
                        converted_state_dict[key.replace("attn.to_q", rename_attn_key)] = to_q
                    else:
                        qkv = torch.cat([to_q, to_k, to_v], dim=0)
                        converted_state_dict[key.replace("attn.to_q", rename_attn_key)] = qkv

        if "add_q_proj" in key:
            to_q = converted_state_dict.pop(key)
            to_k = converted_state_dict.pop(key.replace("add_q_proj", "add_k_proj"))
            to_v = converted_state_dict.pop(key.replace("add_q_proj", "add_v_proj"))
            rename_attn_key = "txt_attn_qkv"
            if "lora_A" in key:
                converted_state_dict[key.replace("attn.add_q_proj", rename_attn_key)] = to_q
            else:
                qkv = torch.cat([to_q, to_k, to_v], dim=0)
                converted_state_dict[key.replace("attn.add_q_proj", rename_attn_key)] = qkv

    for key in list(converted_state_dict.keys()):
        new_key = key[:]
        if "token_refiner" in key and "attn.to_out.0" in new_key:
            new_key = new_key.replace("attn.to_out.0", "self_attn_proj")
        if "token_refiner" in key and "ff" in new_key:
            new_key = new_key.replace("ff", "mlp")
        if "token_refiner" in key and "norm_out.linear" in new_key:
            new_key = new_key.replace("norm_out.linear", "adaLN_modulation.1")
        if "context_embedder" in key and "time_text_embed.text_embedder.linear_1" in new_key:
            new_key = new_key.replace("time_text_embed.text_embedder.linear_1", "c_embedder.linear_1")
        if "context_embedder" in key and "time_text_embed.text_embedder.linear_2" in new_key:
            new_key = new_key.replace("time_text_embed.text_embedder.linear_2", "c_embedder.linear_2")
        if "context_embedder" in key and "time_text_embed.timestep_embedder.linear_1" in new_key:
            new_key = new_key.replace("time_text_embed.timestep_embedder.linear_1", "t_embedder.mlp.0")
        if "context_embedder" in key and "time_text_embed.timestep_embedder.linear_2" in new_key:
            new_key = new_key.replace("time_text_embed.timestep_embedder.linear_2", "t_embedder.mlp.2")
        if "single_transformer_blocks" in key and "proj_out" in new_key:
            new_key = new_key.replace("proj_out", "linear2")
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT_REVERSE.items():
            new_key = new_key.replace(replace_key, rename_key)
        converted_state_dict[new_key] = converted_state_dict.pop(key)

    # Remove "transformer." prefix
    # for key in list(converted_state_dict.keys()):
        # if key.startswith("transformer."):
            # converted_state_dict[key[len("transformer."):]] = converted_state_dict.pop(key)

    # Add back "diffusion_model." prefix
    # for key in list(converted_state_dict.keys()):
        # converted_state_dict[f"diffusion_model.{key}"] = converted_state_dict.pop(key)
    
    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)
    
    for key in list(converted_state_dict.keys()):
        converted_state_dict[key.replace(".transformer_blocks.", ".double_blocks.")] = converted_state_dict.pop(key)
    
    for key in list(converted_state_dict.keys()):
        converted_state_dict[key.replace(".single_transformer_blocks.", ".single_blocks.")] = converted_state_dict.pop(key)
    
    return converted_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.input.endswith(".pt"):
        diffusers_state_dict = torch.load(args.input, map_location="cpu", weights_only=True)
    elif args.input.endswith(".safetensors"):
        diffusers_state_dict = load_file(args.input)

    original_format_state_dict = convert_diffusers_to_hunyuan_video_lora(diffusers_state_dict)
    
    if args.output is not None:
        output_path_or_name = Path(args.output)
        if output_path_or_name.as_posix().endswith(".safetensors"):
            os.makedirs(output_path_or_name.parent, exist_ok=True)
            save_file(original_format_state_dict, output_path_or_name)
        else:
            os.makedirs(output_path_or_name, exist_ok=True)
            output_path_or_name = output_path_or_name / "pytorch_lora_weights.safetensors"
            save_file(original_format_state_dict, output_path_or_name)
    else:
        if args.input.endswith(".safetensors"):
            output_path = args.input.replace(".safetensors", "_converted.safetensors")
        else:
            output_path = args.input + "_converted.safetensors"
        save_file(original_format_state_dict, output_path)