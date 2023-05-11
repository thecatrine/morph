"""Morph

Usage:
    morph.py train [--train-config=<train-config>]
    morph.py fake-train [--train-config=<train-config>]

Options:
    -h --help                       Show this screen.
    -c --train-config FILE          Path to training config file [default: train_config.yaml]
"""

from docopt import docopt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import tqdm
import yaml

import os
import random

import train_utils
import model as m
import model_utils as mu

# Setup for distributed training

MASTER_PORT = str(random.randint(10000, 20000))

def setup_parallel(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['WORLD_SIZE'] = str(world_size)

    print("Initializing process group...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Process group initialized.")

def cleanup_parallel():
    dist.destroy_process_group()


# Move these
C_RUN_NAME = 'run_name'
C_NUM_GPUS = 'num_gpus'

# Training
def train_setup(rank, train_config):
    RAW_LOG_DIR = os.path.join('runs', train_config['setup'][C_RUN_NAME])
    RAW_MODEL_DIR = os.path.join('models', train_config['setup'][C_RUN_NAME])

    LOG_DIR = RAW_LOG_DIR
    MODEL_DIR = RAW_MODEL_DIR
    i = 2
    while os.path.exists(LOG_DIR) or os.path.exists(MODEL_DIR):
        LOG_DIR =f"{RAW_LOG_DIR}_{i}"
        MODEL_DIR = f"{RAW_MODEL_DIR}_{i}"
        i += 1
    
    os.makedirs(LOG_DIR)
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
    NUM_GPUS = world_size
    setup_parallel(rank, world_size)

    if rank == 0:
        writer = train_setup(rank, train_config)

    BATCH_SIZE = train_config['training']['parallelism']['batch_size']
    ACCUMULATION = train_config['training']['parallelism']['accumulate_batches']

    loader_batch_size = BATCH_SIZE // NUM_GPUS // ACCUMULATION

    # Load the dataset and partition it by GPU
    train_dataloader, test_dataloader = train_utils.prepare_dataset(
        rank=rank,
        world_size=world_size,
        batch_size=loader_batch_size,
    )

    # Model and Optimizer
    # TODO: Options here?
    device = torch.device("cuda")

    model = m.get_model(train_config).to(device)
    optimizer_config = train_config['training']['optimizer']

    # Adjust the learning rate for the grad accumulation
    optimizer_config['lr'] /= ACCUMULATION

    optimizer = torch.optim.Adam(model.parameters, **optimizer_config)

    # Scheduler and Scaler
    # TODO: What if train_dataloader doesn't have a clean len for streaming or something

    EPOCHS = train_config['training']['options']['epochs']
    WARMUP_FRACTION = train_config['training']['options']['warmup_frac']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda e: train_utils.scheduler_function(
            e,
            WARMUP_FRACTION,
            loader_batch_size=loader_batch_size,
            epochs=EPOCHS,
            N=len(train_dataloader),
            parallelism=NUM_GPUS,
            accumulation=ACCUMULATION,
        )
    )

    FLOAT_16 = train_config['training']['options']['float16']
    train_dtype = torch.float16 if FLOAT_16 else torch.float32
    scaler = torch.cuda.amp.GradScaler()

    # Setup for parallelism
    print(f"{rank}: Moving model to rank")
    model = model.to(rank).train()

    print(f"{rank}: DDP")
    model = DDP(model, device_ids=[rank])#, output_device=rank)#, find_unused_parameters=True)
    print(f"{rank}: Done with DDP")

    # TODO maybe cleaner here
    total_examples = 0
    lowest_loss = 10e10

    for epoch in range(EPOCHS):
        loader = iter(train_dataloader)

        for i, batch in enumerate(tqdm.tqdm(loader)):
            total_examples += loader_batch_size
            # TODO This is still not general, fix this with a dataset utility or something
            # Or else customize it by hand so it's not clunky
            ###########################
            # CUSTOM LOSS LOGIC :O    #
            ###########################

            x = batch['pixels']
            timesteps = torch.rand((x.shape[0],))

            with torch.autocast(device_type='cuda', dtype=train_dtype):
                loss = mu.denoising_score_estimation(model, x.to(device), timesteps.to(device))
            
            ###########################

            scaler.scale(loss).backward()
            if step % ACCUMULATION == ACCUMULATION - 1:
                scaler.unscale_(optimizer)
                #TODO option for this torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
                scheduler.step()

            if NUM_GPUS > 1:
                # Gather loss from all GPUs
                outputs = [None for _ in range(world_size)]
                dist.all_gather_object(outputs, loss)
            else:
                outputs = [loss]
            
            # Report loss by number of examples seen
            if rank == 0:
                outputs = [o.cpu() for o in outputs]
                loss = torch.mean(torch.stack(outputs))

                writer.add_scalar('Loss/Train', loss, total_examples)
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], total_examples)

            # Test loss
            test_every = train_config['training']['options']['test_every']
            if step % test_every == test_every - 1:
                test_loss = get_test_loss(model, test_dataloader)

                if NUM_GPUS > 1:
                    outputs = [None for _ in range(world_size)]
                    dist.all_gather_object(outputs, test_loss)
                else:
                    outputs = [test_loss]

                if rank == 0:
                    outputs = [o.cpu() for o in outputs]
                    test_loss = torch.mean(torch.stack(outputs))

                    writer.add_scalar('Loss/Test', test_loss, total_examples)

                    if test_loss < lowest_loss:
                        print("Saving best model. Loss=", test_loss)
                        train_utils.save_state(
                            epoch=epoch,
                            test_loss=test_loss,
                            model=model.module, # model.module here because of DPP
                            optimizer=optimizer,
                            scheduler=scheduler,
                            file=os.path.join(train_config['MODEL_DIR'], f'model_best.pt'), 
                        )

    # Save at end of training as well
    if rank == 0:
        train_utils.save_state(
            epoch=epoch,
            test_loss=test_loss,
            model=model.module,
            optimizer=optimizer,
            scheduler=scheduler,
            file=os.path.join(train_config['MODEL_DIR'], f'model_last.pt'), 
        )

    if NUM_GPUS > 1:
        cleanup_parallel()


def train(train_config_path):
    # TODO: Let arguments or env vars shadow config, but remember to clearly signpost it
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)

        NUM_GPUS = train_config['training']['parallelism'][C_NUM_GPUS]
        print("NUM_GPUS", NUM_GPUS)
        if  NUM_GPUS == 1:
            train_internal(rank=0, world_size=1, train_config=train_config)
        else:
            mp.spawn(
                train_internal,
                args=(NUM_GPUS, train_config),
                nprocs=NUM_GPUS,
            )

    

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments)

    if arguments['train']:
        train(arguments['--train-config'])



