# %% 
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import einops
import yaml

import math

import model as m
import model_utils as mu
import train_utils
# %%
# Problem specific imports

from PIL import Image

# %%

device = torch.device('cuda')

CHANNELS = 3
MODEL_FILE = 'models/test_5/model_best.pt'
#MODEL_FILE = 'model_latest_mnist.pth'


# TODO: Save model parameters with the model so we can load it and instantiate the model without loading
# the training config

# %%
# This is an implementation of the sampling algorithm from the paper
# TODO: Put reference here

def continuous_sampling(m, samples, steps):
    dt = -1 / steps

    x = samples.clone()

    for t in tqdm.tqdm(np.linspace(1, 0, steps)):
        
        # DW is normal centered at 0 with std = sqrt(dt)
        dw = ( torch.randn_like(x) * np.sqrt(-dt) ).to(device)

        # Rescale score by 1/ sigmas
        score = m(x, t*torch.ones((samples.shape[0],)).to(device)) / sigma(t)


        gt = diffusion(t)
        dx = -1.0*(gt**2)*score*dt + gt*dw

        #import pdb; pdb.set_trace()

        x = x + dx

        # Corrector?

        # z = torch.randn_like(x)

        # g = m(x, t*torch.ones((samples.shape[0],)).to(device)) / sigma(t)

        # epsilon = 2*(z.norm() / g.norm())**2

        # x = x + epsilon*g + torch.sqrt(2*epsilon)*z

    return x

# Sigma schedule based on timestep
def sigma(t):
    return mu.MIN_SIGMA * (mu.MAX_SIGMA / mu.MIN_SIGMA) ** t

B = np.log(mu.MAX_SIGMA)
A = np.log(mu.MIN_SIGMA)

# Partially calculated a derivative of a term by hand for a gaussian
def dsigmasquared(t):
    return 2*(B-A)*sigma(t)

# Taken from code in paper.
# Had a mismatch between g(t) and g(t)^2 before after rewrite
def diffusion(t):
    s = sigma(t)
    diffusion = s * torch.sqrt(torch.tensor(2 * (B - A),
                                                device=device))
    
    return diffusion


def toimage(foo):
    return Image.fromarray(einops.rearrange(((foo)*256).to(torch.uint8), 'c x y -> x y c').cpu().numpy())


##################################
# CUSTOM INFERENCE LOGIC HERE :O #
##################################

def t_i(i, N=100):
    p = mu.rho
    return (mu.sigma_max**(1/p) + (i/(N-1))*(mu.sigma_min**(1/p) - mu.sigma_max**(1/p)))**p

def sample_edm(model, start, steps=100):
    dt = -1 / steps
    x = start.clone()
    for i in tqdm.tqdm(range(steps)):
        t = t_i(i, steps)
        next_t = t_i(i+1, steps)

        dt = next_t - t
        # I think it just uses sigma = t
        sigma = torch.Tensor([t]).reshape((1, 1, 1, 1)).to(device)
        with torch.no_grad():
        #    D = mu.D_theta(model, sigma, x)
            timesteps = sigma**2 * torch.ones((x.shape[0],)).to(x.device)
            #import ipdb; ipdb.set_trace()
            print(x.shape, timesteps.shape)
            score = model(x, *timesteps) / sigma
        # dsigma/dt = d/dt(t) = 1
        # dx = - sigma'*sigma*score * dt
        dsigmadt = 1
        # score = 

        dx = -1*(sigma)*(score)*dt

        x += dx

        #import ipdb; ipdb.set_trace()

    return x


import loaders.loader_utils as lu

def images_as_grid(end):
    DIM = math.ceil(math.sqrt(end.shape[0]))
    columns = []
    row = []
    for im in end:
        row.append(im)
        if len(row) == DIM:
            columns.append(torch.cat(row, dim=2))
            row = []
    #if row != []:
    #    columns.append(torch.cat(row, dim=2))

    all_ims = (torch.cat(columns, dim=1).cpu())
    return lu.tensor_to_image(all_ims)
    #return Image.fromarray(einops.rearrange((all_ims*256).to(torch.uint8), 'c x y -> x y c').numpy())
    #return Image.fromarray(all_ims.squeeze(0).cpu().numpy()*255.0)

def inference(model, N, train_config, output_filename):
    start = torch.randn((N, train_config['model']['in_channels'], 32, 32)).to(device)*mu.MAX_SIGMA

    end = sample_edm(model, start)

    grid_image = images_as_grid(end)

    grid_image.convert("RGB").save(output_filename)

    return end, grid_image


#Image.fromarray(all_ims, mode='RGB')


# %%

def get_model(model_file, train_config_filename):
    with open(train_config_filename) as f:
            train_config = yaml.safe_load(f)

    model = m.get_model(train_config).to(device)
    saved = torch.load(model_file)
    model.load_state_dict(saved['model'])
    model.eval()

    return model, train_config

def inference_main(model_file, train_config_filename, output_filename="inference.png"):
    model, train_config = get_model(model_file, train_config_filename)

    return inference(model, 16, train_config, output_filename)

# %%