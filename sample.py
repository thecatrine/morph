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

def old_continuous_sampling(m, samples, steps):
    dt = -1 / steps

    x = samples.clone()

    for t in tqdm.tqdm(np.linspace(1, 0, steps)):
        
        # DW is normal centered at 0 with std = sqrt(dt)
        dw = ( torch.randn_like(x) * np.sqrt(-dt) ).to(device)

        # Rescale score by 1/ sigmas
        with torch.no_grad():
            score = m(x, t*torch.ones((samples.shape[0],)).to(device)) / sigma(t)


        gt = dsig(t)
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
# def sigma(t):
#     return mu.MIN_SIGMA * (mu.MAX_SIGMA / mu.MIN_SIGMA) ** t

# B = np.log(mu.MAX_SIGMA)
# A = np.log(mu.MIN_SIGMA)

def toimage(foo):
    return Image.fromarray(einops.rearrange(((foo)*256).to(torch.uint8), 'c x y -> x y c').cpu().numpy())


##################################
# CUSTOM INFERENCE LOGIC HERE :O #
##################################

#def t_i(i, N=100):
#    p = mu.rho
#    return (mu.sigma_max**(1/p) + (i/(N-1))*(mu.sigma_min**(1/p) - mu.sigma_max**(1/p)))**p


def new_ti(i, N=100):
    p = 7
    sigma_min = mu.sigma_min
    sigma_max = mu.sigma_max

    return ( sigma_max**(1/p) + (i/(N-1))*(sigma_min**(1/p) - sigma_max**(1/p)) )**p

def new_sigma(t):
    return t

def sample_edm(model, start, steps=100):
    x = start.clone()
    for i in tqdm.tqdm(range(steps)):
        t = new_ti(i, steps) #t_i(i, steps)
        next_t = new_ti(i+1, steps) #t_i(i+1, steps)

        dt = next_t - t
        # I think it just uses sigma = t
        sigma = torch.Tensor([new_sigma(t)]).reshape((1, 1, 1, 1)).to(device)

        #sigma = new_sigma(t)*torch.ones((x.shape[0],)).to(device)
        with torch.no_grad():
            D = mu.D_theta(model, sigma, x)
        # dsigma/dt = d/dt(t) = 1
        # dx = - sigma'*sigma*score * dt

        score = (D - x) / sigma**2
        
        dsigmadt = 1
        d = -1.0*dsigmadt*sigma*score
        x1 = x + dt*d
        
        # second order correction
        if i < (steps-1):
            dsigmadt1 = 1
            sigma1 = sigma = torch.Tensor([new_sigma(next_t)]).reshape((1, 1, 1, 1)).to(device)
            
            with torch.no_grad():
                D1 = mu.D_theta(model, sigma1, x1)
            dprime = dsigmadt1/sigma1*(x1 - D1)
            x = x + dt*0.5*(d + dprime)
        else:
            x = x1
        #import ipdb; ipdb.set_trace()

    return x

def sample_edm_stochastic(model, start, steps=100, S_churn=1, S_noise=1.01, S_min=0.2, S_max=0.8):
    x = start.clone()
    for i in tqdm.tqdm(range(steps)):
        t = new_ti(i, steps) #t_i(i, steps)
        next_t = new_ti(i+1, steps) #t_i(i+1, steps)

        # I think it just uses sigma = t
        sigma = torch.Tensor([new_sigma(t)]).reshape((1, 1, 1, 1)).to(device)

        # stochastic part I don't understand yet
        epsilon = torch.randn_like(x)*S_noise
        if t >= S_min and t <= S_max:
            gamma_i = min(S_churn/steps, np.sqrt(2) - 1)
        else:
            gamma_i = 0

        t_hat = t + gamma_i*t
        x = x + np.sqrt(t_hat**2 - t**2)*epsilon
        t = t_hat

        dt = next_t - t

        #sigma = new_sigma(t)*torch.ones((x.shape[0],)).to(device)
        with torch.no_grad():
            D = mu.D_theta(model, sigma, x)
        # dsigma/dt = d/dt(t) = 1
        # dx = - sigma'*sigma*score * dt

        score = (D - x) / sigma**2
        
        dsigmadt = 1
        d = -1.0*dsigmadt*sigma*score
        x1 = x + dt*d

        # second order correction
        if i < (steps-1):
            dsigmadt1 = 1
            sigma1 = sigma = torch.Tensor([new_sigma(next_t)]).reshape((1, 1, 1, 1)).to(device)
            
            with torch.no_grad():
                D1 = mu.D_theta(model, sigma1, x1)
            dprime = dsigmadt1/sigma1*(x1 - D1)
            x = x + dt*0.5*(d + dprime)
        else:
            x = x1
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
    start = torch.randn((N, train_config['model']['in_channels'], 28, 28)).to(device)*mu.MAX_SIGMA

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