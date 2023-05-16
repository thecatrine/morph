import torchvision
import numpy as np
import torch
import torch.nn.functional as F
import math
from . import whiten

def linear_beta_schedule(max_ts, min_beta=0.0001, max_beta=0.02):
    return torch.linspace(min_beta, max_beta, max_ts)

def cosine_beta_schedule(max_ts, s=0.008):
    # from https://openreview.net/forum?id=-NEXDKk8gZ
    x = torch.linspace(0, max_ts, max_ts+1)
    alphas_cumprod = torch.cos(((x / max_ts) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

MAX_TS = 1000
BETAS=cosine_beta_schedule(MAX_TS)
BETAS = torch.tensor(BETAS)
ALPHAS = 1-BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, dim=0)

# THis shit is just copieed from dalle2_pytorch.py:446
ALPHAS_CUMPROD_PREV = F.pad(ALPHAS_CUMPROD[:-1], (1, 0), value = 1.)
POSTERIOR_MEAN_COEF1 = BETAS * torch.sqrt(ALPHAS_CUMPROD_PREV) / (1. - ALPHAS_CUMPROD)
POSTERIOR_MEAN_COEF2 = (1. - ALPHAS_CUMPROD_PREV) * torch.sqrt(ALPHAS) / (1. - ALPHAS_CUMPROD)
POSTERIOR_VARIANCE = BETAS * (1. - ALPHAS_CUMPROD_PREV) / (1. - ALPHAS_CUMPROD)
POSTERIOR_LOG_VARIANCE = torch.log(POSTERIOR_VARIANCE.clamp(min=1e-20))

def noise_img(img, n=1):
    assert n > 0
    return torch.normal(np.sqrt(ALPHAS_CUMPROD[n-1])*img, np.sqrt(1-ALPHAS_CUMPROD[n-1]))

def weighted_timestep(max_ts=1000):
    return math.floor(np.random.random() * max_ts)

CLIP_GUIDANCE_SCALE = 150
def take_step(noised_img, predicted_true_img, n, condition=None):
    mean = POSTERIOR_MEAN_COEF1[n-1] * predicted_true_img + POSTERIOR_MEAN_COEF2[n-1] * noised_img
    noise = torch.randn_like(mean)
    # TODO: Figure out whether we fucked up the logic with this zero indexing thing
    # nonzero_mask = (1 - (n == 0).float()).reshape(mean.shape[0], *((1,) * (len(mean.shape) - 1)))
    if condition is not None:
        blah = CLIP_GUIDANCE_SCALE * condition
        print(noise.norm(), blah.norm())
        noise = blah*(0.1**0.5) + noise*(0.9**0.5)

    return mean + (0.5 * POSTERIOR_LOG_VARIANCE[n-1]).exp() * noise

SCALE = 10
whitener = whiten.Whitener()
def image_to_tensor(im):
    tensor = torchvision.transforms.ToTensor()(im)
    if tensor.shape[0] == 4:
        tensor = tensor[:3, :, :]

    scaled_tensor = (tensor - 0.5) * SCALE
    return whitener.transform(scaled_tensor)

def tensor_to_image(tensor):
    tensor = whitener.untransform(tensor)
    tensor = (tensor / SCALE) + 0.5
    tensor = torch.clip(input=tensor, min=0, max=1)

    image = torchvision.transforms.ToPILImage()(tensor)
    return image

# Not actually whitening, just normalizing
def unwhiten(tensor):
    return whitener.untransform(tensor)

def unwhiten_differentiable(tensor):
    return whitener.differentiable_untransform(tensor)
