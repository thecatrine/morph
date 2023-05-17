import math
from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Taken from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
# My utils

class InvokeFunction(nn.Module):
    def __init__(self, function, *a, **kw):
        super().__init__()
        self.a = a
        self.kw = kw
        self.function = function

    def forward(self, *a, **kw):
        return self.function(*(a + self.a), **{**kw, **self.kw})
    

# Parameters from Karras et al 2022

sigma_min = 0.002
sigma_max = 80
sigma_data = 0.5
rho = 7
P_mean = -1.2
P_std = 1.2

def sigma(t):
    return t

def c_skip(sigma):
    return (sigma_data**2) / (sigma**2 + sigma_data**2)

def c_out(sigma):
    return (sigma*sigma_data) / th.sqrt(sigma_data**2 + sigma**2)

def c_in(sigma):
    return 1 / th.sqrt(sigma_data**2 + sigma**2)

def c_noise(sigma):
    return 0.25*th.log(sigma)

def noise_distribution(shape):
    noise = th.randn(shape) * P_std + P_mean

    return th.exp(noise)

def loss_weighting(sigma):
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2


def D_theta(model, sigma, x):
    cond_x = c_in(sigma) * x
    cond_sigma = c_noise(sigma)

    #import ipdb; ipdb.set_trace()
    return c_skip(sigma) * x + c_out(sigma)*model(cond_x, cond_sigma.reshape((-1,)))

def new_denoising_score_estimation(score_net, samples, sigmas):
    #reshaped_sigmas = sigmas.reshape(samples.shape[0], 1, 1, 1)

    #import ipdb; ipdb.set_trace()
    z = th.randn_like(samples)
    noise = z*sigmas

    scores = D_theta(score_net, sigmas, samples + noise)

    loss = th.square(scores - samples)

    loss = loss_weighting(sigmas) * loss

    return loss.mean()

# Implementation of loss for denoising score estimation
# TODO: Find reference for this in the paper

# I think 50 is too high for MNIST but these settings were for emoji, and I wanted a reference that you could
# get from datasets. Bump this down?

#MAX_SIGMA = 50
MAX_SIGMA = 1
MIN_SIGMA = 0.01
def sigma(t):
    B = np.log(MAX_SIGMA)
    A = np.log(MIN_SIGMA)

    C = (B-A)*t + A

    return th.exp(C)

def denoising_score_estimation(score_net, samples, timesteps):
    sigmas = sigma(timesteps)
    #import ipdb; ipdb.set_trace()

    reshaped_sigmas = sigmas.reshape(samples.shape[0], 1, 1, 1)

    z = th.randn_like(samples)
    noise = z*reshaped_sigmas
    
    # Rescale output of score net by 1/sigma
    scores = score_net(samples + noise, timesteps) / reshaped_sigmas

    loss = 0.5 * th.square(scores*reshaped_sigmas + z)

    return loss.mean()