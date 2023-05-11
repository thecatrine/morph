"""Morph

Usage:
    morph.py train [--train-config=<train-config>]
    morph.py fake-train [--bar=<bar>]

Options:
    -h --help                       Show this screen.
    -c --train-config FILE          Path to training config file [default: train_config.yaml]
"""

from docopt import docopt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import yaml

import os
import random

# Setup for distributed training

def setup_parallel(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_parallel():
    dist.destroy_process_group()


# Move these
RUN_NAME = 'run_name'
NUM_GPUS = 'num_gpus'

# Training
def train_setup(rank, train_config):
    LOG_DIR = os.path.join('runs', train_config[RUN_NAME])
    i = 2
    while os.path.exists(LOG_DIR):
        LOG_DIR += f"_{i}"
        i += 1
    
    os.makedirs(LOG_DIR)

    MODEL_DIR = os.path.join('models', train_config[RUN_NAME])
    os.makedirs(MODEL_DIR)

    train_config.update({
        'LOG_DIR': LOG_DIR,
        'MODEL_DIR': MODEL_DIR
    })

    # Setup tensorboard
    writer = SummaryWriter(log_dir=LOG_DIR)

    return writer

def train_internal(rank, world_size, train_config):
    # Create directories and setup tensorboard
    if rank == 0:
        writer = train_setup(rank, train_config)

    BATCH_SIZE = train_config['training']['parallelism']['batch_size']
    NUM_GPUS = train_config['training']['parallelism']['num_gpus']
    ACCUMULATION = train_config['training']['parallelism']['accumulate_batches']

    loader_batch_size = BATCH_SIZE // NUM_GPUS // ACCUMULATION

    # RSI actually load the dataloaders


def train(train_config_path):
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)

        if train_config[NUM_GPUS] == 1:
            train_internal(rank=0, world_size=1, train_config=train_config)
        else:
            mp.spawn(
                train_internal,
                args=(train_config[NUM_GPUS], train_config),
                nprocs=train_config[NUM_GPUS],
            )

    

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments)

    if arguments['train']:
        train(arguments['--train-config'])



