from datasets import load_dataset

import numpy as np
import torch

import loaders.datasets as ds
import loaders.loader_utils as utils

##############################################################################
# Rewrite this to return your actual dataset and do any prep work required   #
##############################################################################

def map_func(examples):
    #import ipdb; ipdb.set_trace()
    examples['pixels'] = []
    for ex in examples['image']:
        im = np.array(ex)
        tensor = torch.Tensor(im)
        normalized_tensor = (tensor) / 256.0
        examples['pixels'].append(normalized_tensor)
        
    return examples


# def get_datasets():
#     dataset = load_dataset("mnist")

#     train_dataset = dataset['train'].map(map_func, batched=True).with_format('torch')
#     test_dataset = dataset['test'].map(map_func, batched=True).with_format('torch')

#     return {
#         'train': train_dataset,
#         'test': test_dataset,
#     }

def get_datasets():
    train_tensors = torch.load("loaders/twitch/train.pt")
    val_tensors = torch.load("loaders/twitch/test.pt")

    return {
        'train': train_tensors,
        'test': val_tensors,
    }