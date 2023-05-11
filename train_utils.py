from datasets.distributed import split_dataset_by_node

import torch
from torch.utils.data.dataloader import DataLoader

import math

# RSI double check this
def scheduler_function(step, warmup_frac, loader_batch_size, epochs, N, parallelism, accumulation):
        single_gpu_batch = loader_batch_size * accumulation
        single_gpu_examples = epochs * N / parallelism
        total_steps = math.ceil(single_gpu_examples / single_gpu_batch)

        frac = step / total_steps

        if frac < warmup_frac:
            return frac / warmup_frac
        else:
            return 1.0 - ((frac - warmup_frac) / (1.0 - warmup_frac))

def prepare_dataset(
    rank, 
    world_size, 
    batch_size, 
    pin_memory=False, 
    num_workers=0
):
    dataset = dataset.get_datasets()

    dataset_train = split_dataset_by_node(dataset['train'], rank, world_size)
    dataset_test = split_dataset_by_node(dataset['test'], rank, world_size)

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=None,
    )
    
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=None,
    )

    return train_dataloader, test_dataloader


def save_state(epoch, test_loss, model, optimizer, scheduler, file):
    arrs = {
        'epoch': epoch,
        'best_test_loss': test_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    torch.save(arrs, file)

def load_state(model, optimizer, filename):
    loaded = torch.load(filename)

    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    

