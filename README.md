# HunyuanVideo-Training

This is not intended to be a comprehensive all-in-one trainer, it's just the bare minimum framework to support simple one-file training scripts, in the spirit of diffusers example training scripts. It's meant to be easy to read and modify, without too much abstraction getting in the way, but with enough optimization to not require rental cloud compute. The default configuration uses < 16 GB of VRAM (although < 24 GB is the target), and runs natively on windows. To achieve this, the diffusion model is quantized to nf4 using bitsandbytes (similar to QLoRA), and text embeddings are pre-computed. Latents are encoded on the fly however, to reduce overfitting and make it easier to work with larger datasets.

Don't expect much support, as this is primarily for my own use, but I'm sharing it for others who want to tinker with training code, and because I was frustrated with diffusers' switch with recent video models from single file training scripts to finetrainers. Any code written by me is released under MIT license (aka do whatever you want), but the HunyuanVideo model is subject to the [tencent community license](https://github.com/Tencent/HunyuanVideo/blob/main/LICENSE.txt).

## Install

```
sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg

git clone https://github.com/svjack/HunyuanVideo-Training
cd HunyuanVideo-Training
pip install torch torchvision
pip install -r requirements.txt

#python -m venv .venv
#.venv\Scripts\activate

#pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

(or follow whatever approach you prefer for environment management and pytorch/triton/etc)

```
pip install -r requirements.txt
```

## Before training

Activate the venv (you can use launch_venv.bat if on windows)

Download models:

```
python train_hunyuan_lora.py --download_model
```

or if you want to train skyreels i2v:

```
python train_hunyuan_lora.py --download_model --skyreels_i2v
```

(if you already have the diffusers models saved elsewhere, you can skip downloading and train with `--pretrained_model` pointing to the correct folder)

Expected folder structure:
```
HunyuanVideo-Training
  /models
    /scheduler
    /text_encoder
    /text_encoder_2
    /tokenizer
    /tokenizer_2
    /transformer
    /transformer-skyreels-i2v
    /vae
```

## Dataset prep

Your dataset should be structured something like this:

```
dataset
  /train
    /subfolder
      sample1.mp4
      sample1.txt
    sample2.jpg
    sample2.txt
    ...
  /val
    ...
```
Training data goes in `/train`, validation data goes in `/val`, `/validation`, or `/test`. Subfolders within the train or val folder are scanned recursively, so organize them however you like.

If no validation set is provided, then validation will fall back to the training set. This is NOT RECOMMENDED, it will make it impossible to judge overfitting from the validation loss.

Image and video files are supported, and captions should be the same filename as the media file but with .txt extension. See [here](https://github.com/spacepxl/HunyuanVideo-Training/blob/main/train_hunyuan_lora.py#L43) for the list of filetypes, although not all are tested, and others might work just by adding them to the list (anything supported by decord for videos or Pillow for images)

Once you have a dataset, cache the text embeddings:

```
python train_hunyuan_lora.py --dataset "example/dataset" --cache_embeddings
```

## Training

```
python train_hunyuan_lora.py --dataset "example/dataset"
```

All other arguments are optional, the defaults should be a reasonable starting point.

By default, resolutions are randomized in buckets, frame length is set based on a context length budget (so, semi-random but keeping similar memory and compute cost), and the start frame is randomly chosen from the range allowed by frame length. The default `--token_limit` of 10000 is good for < 16 GB, and about 30s/step on a 3090. Raising the token limit will use more memory and more time per step, and increase the number of frames at every resolution.

Optionally you can set `--resolution` to disable the resolution buckets and use a square crop of that exact size. If you set `--num_frames` it will use that as the upper limit (some samples may use fewer frames depending on resolution, or limited by short clips/images).

For image to video training, use `--skyreels_i2v` to load the skyreels model and use the first frame as image conditioning.

Warped noise from [Go With The Flow](https://eyeline-research.github.io/Go-with-the-Flow/) can be enabled by `--warped_noise`, note that this will take longer to adapt to than normal random noise, so it's more for general adapter training than character/style loras, and you should use a large dataset, ideally larger than the number of training steps.

## Loss
```python
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import time
from IPython.display import clear_output
import pandas as pd

# 定义事件文件路径
event_file_path = "outputs/2025-02-25_13-42-37/events.out.tfevents.1740490957.featurize.119960.0"

# 创建 EventAccumulator 对象
event_acc = event_accumulator.EventAccumulator(event_file_path)

# 加载事件数据
event_acc.Reload()

# 定义窗口大小
window_size = 10  # 可以根据需要调整窗口大小

# 主循环，每秒刷新一次
while True:
    # 清除之前的输出
    clear_output(wait=True)
    
    # 重新加载事件数据
    event_acc.Reload()
    
    # 获取所有标签
    tags = event_acc.Tags()
    
    # 检查是否存在 'loss/train' 数据
    if 'scalars' in tags and 'loss/train' in tags['scalars']:
        # 提取 'loss/train' 数据
        train_loss_events = event_acc.Scalars('loss/train')
        
        # 提取步骤和损失值
        steps = [event.step for event in train_loss_events]
        values = [event.value for event in train_loss_events]
        
        # 将数据转换为 Pandas DataFrame
        df = pd.DataFrame({'steps': steps, 'values': values})
        
        # 计算滑动窗口均值
        df['smoothed_values'] = df['values'].rolling(window=window_size, min_periods=1).mean()
        
        # 绘制原始 loss/train 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(df['steps'], df['values'], label='Train Loss (Raw)', color='blue', alpha=0.3)
        
        # 绘制平滑后的 loss/train 曲线
        plt.plot(df['steps'], df['smoothed_values'], label=f'Train Loss (Smoothed, Window={window_size})', color='red')
        
        plt.title('Training Loss Over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No 'loss/train' data found in the event file.")
    
    # 等待 1 秒
    time.sleep(1)
```

## Inference
- Girl Demo

```bash
python test_hunyuan_lora.py --lora="hunyuan_pixel_lora_outputs/outputs/2025-02-25_13-42-37/checkpoints/hyv-lora-00000600.safetensors" --prompt="The video showcases a pixel art animation featuring charming anime-style scene featuring a pink-haired girl with angel wings. She's seated at a desk, enjoying a donut while working on a laptop. The setting is a cozy, pastel-colored room with a pink chair, a milk carton, and a coffee cup. The girl's expression is one of delight as she savors her treat." --alpha=16
```

without lora

https://github.com/user-attachments/assets/9113008e-850a-43c3-896c-97bc5053a423

with lora

https://github.com/user-attachments/assets/6ba0236c-35ae-4367-b8ea-71ab287a4e70

- Snow hill Demo

```bash
python test_hunyuan_lora.py --lora="hunyuan_pixel_lora_outputs/outputs/2025-02-25_13-42-37/checkpoints/hyv-lora-00000600.safetensors" --prompt="The video showcases a pixel art animation featuring a serene and majestic snowy mountain landscape. The scene is dominated by towering peaks covered in pristine white snow, with a soft gradient of blue and purple hues in the sky. A small cabin with a smoking chimney sits at the base of the mountain, surrounded by pine trees dusted with snow. A winding path leads up the mountain, with footprints visible in the snow. The atmosphere is calm and peaceful, evoking a sense of solitude and wonder." --alpha=16
```

without lora

https://github.com/user-attachments/assets/f3806a62-caa0-416b-97f9-b7cae689d818

with lora

https://github.com/user-attachments/assets/1d7dbd41-c036-4a57-bd90-2f7341623915


## After training

The saved lora checkpoints are in diffusers format, so if you want to use them with the original tencent model (or in comfyui), you'll need to convert them:

```
python convert_diffusers_lora_to_original.py --input_lora "./outputs/example/checkpoints/hyv-lora-00001000.safetensors"
```

You can optionally convert the lora dtype to bf16 or fp16 to save file size. If you set alpha to anything other than rank during training, you'll need to manually input the alpha during conversion with `--alpha`
