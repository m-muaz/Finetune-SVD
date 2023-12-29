import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import PIL

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers

from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# from models.unet_3d_condition import UNet3DConditionModel
# from diffusers.models import AutoencoderKL
# from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset
from einops import rearrange, repeat
# from utils.lora_handler import LoraHandler, LORA_VERSIONS

from PIL import Image
import random
from diffusers import AutoencoderKLTemporalDecoder
from diffusers import UNetSpatioTemporalConditionModel
from diffusers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from utils.dataset import VideoCSVDataset
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
from diffusers.utils.import_utils import is_xformers_available


already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_primary_models(pretrained_model_path):
    # noise_scheduler = DDPMScheduler.from_config(pretrained_model_path, subfolder="scheduler")
    # tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    # unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    noise_scheduler = EulerDiscreteScheduler.from_config(pretrained_model_path, subfolder="scheduler", variant="fp16")
    # tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer") # what tokenizer?
    tokenizer = None
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder", variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_path, subfolder="vae", variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", variant="fp16")
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor", variant="fp16")

    return noise_scheduler, tokenizer, image_encoder, vae, unet, feature_extractor

def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def encode_image(image_processor, feature_extractor, image_encoder, image, device, num_videos_per_prompt, do_classifier_free_guidance):
    dtype = next(image_encoder.parameters()).dtype

    # image = image.detach().cpu().numpy()
    # image = image_processor.pt_to_numpy(image)

    # if not isinstance(image, torch.Tensor):
    #     image = image_processor.pil_to_numpy(image)
    #     image = image_processor.numpy_to_pt(image)

    #     # We normalize the image before resizing to match with the original implementation.
    #     # Then we unnormalize it after resizing.
    #     image = image * 2.0 - 1.0
    #     image = _resize_with_antialiasing(image, (224, 224))
    #     image = (image + 1.0) / 2.0

    #     # Normalize the image with for CLIP input
    #     image = feature_extractor(
    #         images=image,
    #         do_normalize=True,
    #         do_center_crop=False,
    #         do_resize=False,
    #         do_rescale=False,
    #         return_tensors="pt",
    #     ).pixel_values


    # We normalize the image before resizing to match with the original implementation.
    # Then we unnormalize it after resizing.
    image = image * 2.0 - 1.0
    image = _resize_with_antialiasing(image, (224, 224))
    image = (image + 1.0) / 2.0

    # Normalize the image with for CLIP input
    image = feature_extractor(
        images=image,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    image = image.to(device=device, dtype=dtype)
    image_embeddings = image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        negative_image_embeddings = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

    return image_embeddings

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    # extra_params = extra_params if len(extra_params.keys()) > 0 else None
    extra_params = None
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }
    

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list): 
            params = create_optim_params(
                params=itertools.chain(*model), 
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue
            
        if is_lora and  condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)
    
    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = rearrange(latents, "b c f h w -> b f c h w")
    latents = latents * 0.18215

    return latents

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1)  \
    and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        accelerator, 
        unet, 
        image_encoder, 
        vae, 
        output_dir,
        is_checkpoint=False,
        save_pretrained_model=True
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, image_encoder.dtype, vae.dtype 

   # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_save = copy.deepcopy(unet.cpu())
    image_encoder_save = copy.deepcopy(image_encoder.cpu())

    unet_out = copy.deepcopy(accelerator.unwrap_model(unet_save, keep_fp32_wrapper=False))
    image_encoder_out = copy.deepcopy(accelerator.unwrap_model(image_encoder_save, keep_fp32_wrapper=False))

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        path,
        unet=unet_out,
        image_encoder=image_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)
    
    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, image_encoder = accelerator.prepare(unet, image_encoder)
        models_to_cast_back = [(unet, u_dtype), (image_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")
    
    del pipeline
    del unet_out
    del image_encoder_out
    torch.cuda.empty_cache()
    gc.collect()


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 


def encode_vae_image(
    vae,
    image: torch.Tensor,
    device,
    num_videos_per_prompt,
    do_classifier_free_guidance,
):
    image = image.to(device=device)
    image_latents = vae.encode(image).latent_dist.mode()

    if do_classifier_free_guidance:
        negative_image_latents = torch.zeros_like(image_latents)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_latents = torch.cat([negative_image_latents, image_latents])

    # duplicate image_latents for each generation per prompt, using mps friendly method
    image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

    return image_latents

def get_add_time_ids(
    unet,
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
    num_videos_per_prompt,
    do_classifier_free_guidance,
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

    if do_classifier_free_guidance:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

    return add_time_ids

def prepare_latents(
    vae_scale_factor,
    scheduler,
    batch_size,
    num_frames,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
    generator,
    latents=None,
):
    shape = (
        batch_size,
        num_frames,
        num_channels_latents // 2,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    shuffle: bool = True,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = None, # Eg: ("attn1", "attn2")
    extra_unet_params = None,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    resume_step: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    save_pretrained_model: bool = True,
    logger_type: str = 'tensorboard',
    **kwargs
):

    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
       output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models.
    scheduler, tokenizer, image_encoder, vae, unet, feature_extractor = load_primary_models(pretrained_model_path)

    # Freeze any necessary models
    # freeze_models([vae, image_encoder, unet])
    freeze_models([vae, image_encoder])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)
    torch.cuda.empty_cache()
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    trainable_modules_available = trainable_modules is not None
    
    optim_params = [param_optim(unet, trainable_modules_available, extra_params=extra_unet_params)]

    params = create_optimizer_params(optim_params, learning_rate)
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    tokenizer = CLIPTokenizer.from_pretrained("damo-vilab/text-to-video-ms-1.7b", subfolder="tokenizer")
    train_dataset = VideoCSVDataset(csv_path='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/wangqiang121/animating/svd/Video-BLIP2-Preprocessor-main/train_data/validation_warping.csv',
                              tokenizer=tokenizer,
                              fps = 29,
                              n_sample_frames=25,
                              width=1024,
                              height=576)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=shuffle
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, image_encoder = accelerator.prepare(
        unet, 
        optimizer, 
        train_dataloader, 
        lr_scheduler, 
        image_encoder
    )

    # Enable VAE slicing to save memory.
    # vae.enable_slicing()

    # For mixed precision training we cast the image_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)
    num_videos_per_prompt = 1
    fps = 7
    generator = torch.manual_seed(42)
    noise_aug_strength = 0.02
    motion_bucket_id = 127
    num_inference_steps = 25
    min_guidance_scale = 1.0

    # Move text encoders, and VAE to GPU
    models_to_cast = [image_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet):

                with accelerator.autocast():

                    # image = Image.fromarray(batch["image"].detach().cpu().numpy())
                    image = batch["image"]
                    max_guidance_scale = 3.0

                    # 0. Default height and width to unet
                    height = train_data.height or unet.config.sample_size * vae_scale_factor
                    width = train_data.width or unet.config.sample_size * vae_scale_factor

                    num_frames = train_data.num_frames if train_data.num_frames is not None else unet.config.num_frames
                    decode_chunk_size = None
                    decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

                    # 2. Define call parameters
                    # if isinstance(Image.fromarray(image), PIL.Image.Image):
                    #     batch_size = 1
                    # elif isinstance(image, list):
                    #     batch_size = len(image)
                    # else:
                    #     batch_size = image.size[0]
                    batch_size = image.size()[0]
                    image = image[0]
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                    # corresponds to doing no classifier free guidance.
                    do_classifier_free_guidance = max_guidance_scale > 1.0

                    # 3. Encode input image
                    image_embeddings = encode_image(image_processor, feature_extractor, image_encoder, image, device, num_videos_per_prompt, do_classifier_free_guidance)

                    # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
                    # is why it is reduced here.
                    # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
                    fps = fps - 1

                    # 4. Encode input image using VAE
                    image = image_processor.preprocess(image, height=height, width=width)
                    noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
                    image = image + noise_aug_strength * noise

                    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
                    if needs_upcasting:
                        vae.to(dtype=torch.float32)

                    image_latents = encode_vae_image(vae, image, device, num_videos_per_prompt, do_classifier_free_guidance)
                    image_latents = image_latents.to(image_embeddings.dtype)

                    # cast back to fp16 if needed
                    if needs_upcasting:
                        vae.to(dtype=torch.float16)

                    # Repeat the image latents for each frame so we can concatenate them with the noise
                    # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
                    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

                    # 5. Get Added Time IDs
                    added_time_ids = get_add_time_ids(
                        unet,
                        fps,
                        motion_bucket_id,
                        noise_aug_strength,
                        image_embeddings.dtype,
                        batch_size,
                        num_videos_per_prompt,
                        do_classifier_free_guidance,
                    )
                    added_time_ids = added_time_ids.to(device)

                    # 4. Prepare timesteps
                    # scheduler.set_timesteps(num_inference_steps, device=device)
                    # timesteps = scheduler.timesteps
                    # Sample a random timestep for each video
                    pixel_values = batch["pixel_values"]
                    # pixel_values
                    latents = tensor_to_vae_latent(pixel_values, vae)
                    bsz = latents.shape[0]
                    # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    # timesteps = timesteps.long()
                    scheduler.set_timesteps(num_inference_steps, device=device)
                    random_integer = random.randint(0, num_inference_steps)
                    timesteps = scheduler.timesteps
                    timesteps = timesteps[random_integer]
                    

                    # 5. Prepare latent variables
                    # num_channels_latents = unet.config.in_channels
                    # latents = prepare_latents(
                    #     vae_scale_factor,
                    #     scheduler,
                    #     batch_size * num_videos_per_prompt,
                    #     num_frames,
                    #     num_channels_latents,
                    #     height,
                    #     width,
                    #     image_embeddings.dtype,
                    #     device,
                    #     generator,
                    #     latents,
                    # )

                    # 7. Prepare guidance scale
                    guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
                    guidance_scale = guidance_scale.to(device, latents.dtype)
                    guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
                    guidance_scale = _append_dims(guidance_scale, latents.ndim)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)

                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    # predict the noise residual
                    # noise_pred = unet(
                    #     latent_model_input,
                    #     timesteps,
                    #     encoder_hidden_states=image_embeddings,
                    #     added_time_ids=added_time_ids,
                    #     return_dict=False,
                    # )[0]

                    # Get the target for loss depending on the prediction type
                    if scheduler.prediction_type == "epsilon":
                        target = noise
                    elif scheduler.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {scheduler.prediction_type}")
                    
                    # Get video length
                    video_length = latents.shape[2]

                    # Here we do two passes for video and text training.
                    # If we are on the second iteration of the loop, get one frame.
                    # This allows us to train text information only on the spatial layers.
                    losses = []
                    should_truncate_video = video_length > 1

                    # We detach the encoder hidden states for the first pass (video frames > 1)
                    # Then we make a clone of the initial state to ensure we can train it in the loop.
                    detached_encoder_state = encoder_hidden_states.clone().detach()
                    trainable_encoder_state = encoder_hidden_states.clone()

                    for i in range(2):

                        should_detach = noisy_latents.shape[2] > 1 and i == 0

                        if should_truncate_video and i == 1:
                            noisy_latents = noisy_latents[:,:,1,:,:].unsqueeze(2)
                            target = target[:,:,1,:,:].unsqueeze(2)
                                
                        encoder_hidden_states = (
                            detached_encoder_state if should_detach else trainable_encoder_state
                        )

                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        losses.append(loss)
                        
                        # This was most likely single frame training or a single image.
                        if video_length == 1 and i == 0: break

                    loss = losses[0] if len(losses) == 1 else losses[0] + losses[1] 
                
                # with accelerator.autocast():
                #     loss, latents = finetune_unet(batch, train_encoder=train_text_encoder)
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                try:
                    accelerator.backward(loss)
                    params_to_clip = unet.parameters()

                    if max_grad_norm > 0:
                        if accelerator.sync_gradients:
                            params_to_clip = list(unet.parameters())
                                
                            accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                            
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}") 
                    continue

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            
                if global_step % checkpointing_steps == 0:
                    save_pipe(
                        pretrained_model_path, 
                        global_step, 
                        accelerator, 
                        unet, 
                        image_encoder, 
                        vae, 
                        output_dir,
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    if global_step == 1: print("Performing validation prompt.")
                    if accelerator.is_main_process:
                        
                        with accelerator.autocast():
                            unet.eval()
                            image_encoder.eval()

                            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                                pretrained_model_path,
                                image_encoder=image_encoder,
                                vae=vae,
                                unet=unet
                            )

                            diffusion_scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler

                            # prompt = text_prompt if len(validation_data.prompt) <= 0 else validation_data.prompt
                            image = load_image("./datasets/02.jpeg")

                            curr_dataset_name = batch['dataset']
                            save_filename = f"{global_step}_dataset-{curr_dataset_name}"

                            out_file = f"./output_samples/{save_filename}.mp4"
                            
                            with torch.no_grad():
                                # video_frames = pipeline(
                                #     image,
                                #     width=validation_data.width,
                                #     height=validation_data.height,
                                #     num_frames=validation_data.num_frames,
                                #     num_inference_steps=validation_data.num_inference_steps,
                                #     guidance_scale=validation_data.guidance_scale
                                # ).frames
                                video_frames = pipeline(image, decode_chunk_size=8).frames[0]
                            export_to_video(video_frames, out_file, fps=7)

                            del pipeline
                            torch.cuda.empty_cache()

                    logger.info(f"Saved a new sample to {out_file}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
                pretrained_model_path, 
                global_step, 
                accelerator, 
                unet, 
                image_encoder, 
                vae, 
                output_dir,
                is_checkpoint=False,
                save_pretrained_model=save_pretrained_model
        )     
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/svd.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
