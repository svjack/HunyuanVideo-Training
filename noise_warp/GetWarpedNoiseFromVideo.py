import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from .noise_warp import NoiseWarper, mix_new_noise
from .raft import RaftOpticalFlow

def get_downtemp_noise(noise, noise_downtemp_interp, interp_to=13):   
    if noise_downtemp_interp == 'nearest':
        return resize_list(noise, interp_to)
    elif noise_downtemp_interp == 'blend':
        return downsamp_mean(noise, interp_to)
    elif noise_downtemp_interp == 'blend_norm':
        return normalized_noises(downsamp_mean(noise, interp_to))
    elif noise_downtemp_interp == 'randn':
        return torch.randn_like(resize_list(noise, interp_to))
    else:
        return noise

def downsamp_mean(x, l=13):
    return torch.stack([sum(u) / len(u) for u in split_into_n_sublists(x, l)])

def normalized_noises(noises):
    #Noises is in TCHW form
    return torch.stack([x / x.std(1, keepdim=True) for x in noises])

def resize_list(array:list, length: int):
    assert isinstance(length, int), "Length must be an integer, but got %s instead"%repr(type(length))
    assert length >= 0, "Length must be a non-negative integer, but got %i instead"%length

    if len(array) > 1 and length > 1:
        step = (len(array) - 1) / (length - 1)
    else:
        step = 0  # default step size to 0 if array has only 1 element or target length is 1
        
    indices = [round(i * step) for i in range(length)]
    
    if isinstance(array, np.ndarray) or isinstance(array, torch.Tensor):
        return array[indices]
    else:
        return [array[i] for i in indices]
    
def split_into_n_sublists(l, n):
    if n <= 0:
        raise ValueError("n must be greater than 0 but n is "+str(n))

    if isinstance(l, str):
        return ''.join(split_into_n_sublists(list(l), n))

    L = len(l)
    indices = [int(i * L / n) for i in range(n + 1)]
    return [l[indices[i]:indices[i + 1]] for i in range(n)]


class GetWarpedNoiseFromVideo:
    def __init__(self, raft_size="large", device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.raft_model = RaftOpticalFlow(version=raft_size, device=self.device, dtype=self.dtype)
    
    def __call__(
        self,
        images,
        degradation = 0.0,
        noise_channels = 4,
        spatial_downscale_factor = 8,
        target_latent_count = 16,
        noise_downtemp_interp = "nearest",
    ):
        resize_flow = 1
        resize_frames = 1
        downscale_factor = round(resize_frames * resize_flow) * spatial_downscale_factor
    
        # Load video frames into a [B, C, H, W] tensor, where C=3 and values are between -1 and 1
        B, C, H, W = images.shape
        video_frames = images
        
        def downscale_noise(noise):
            down_noise = F.interpolate(noise, scale_factor=1/downscale_factor, mode='area')  # Avg pooling
            down_noise = down_noise * downscale_factor #Adjust for STD
            return down_noise

        warper = NoiseWarper(
            c = noise_channels,
            h = resize_flow * H,
            w = resize_flow * W,
            device = self.device,
            post_noise_alpha = 0,
            progressive_noise_alpha = 0,
        )

        prev_video_frame = video_frames[0]
        noise = warper.noise

        down_noise = downscale_noise(noise)
        numpy_noise = down_noise.cpu().numpy().astype(np.float16) # In HWC form. Using float16 to save RAM, but it might cause problems on come CPU

        numpy_noises = [numpy_noise]
        # for video_frame in tqdm(video_frames[1:], desc="Calculating noise warp", leave=False):
        for video_frame in video_frames[1:]:
            dx, dy = self.raft_model(prev_video_frame, video_frame)
            noise = warper(dx, dy).noise
            prev_video_frame = video_frame

            numpy_flow = np.stack(
                [
                    dx.cpu().numpy().astype(np.float16),
                    dy.cpu().numpy().astype(np.float16),
                ]
            )
            down_noise = downscale_noise(noise)
            numpy_noise = down_noise.cpu().numpy().astype(np.float16)
            numpy_noises.append(numpy_noise)
        
        numpy_noises = np.stack(numpy_noises).astype(np.float16)
        noise_tensor = torch.from_numpy(numpy_noises).squeeze(1).cpu().float()

        downtemp_noise_tensor = get_downtemp_noise(
            noise_tensor,
            noise_downtemp_interp = noise_downtemp_interp,
            interp_to = target_latent_count,
        ) # B, F, C, H, W
        downtemp_noise_tensor = downtemp_noise_tensor[None]
        downtemp_noise_tensor = mix_new_noise(downtemp_noise_tensor, degradation)
        downtemp_noise_tensor = downtemp_noise_tensor.squeeze(0)

        return downtemp_noise_tensor # BCHW?