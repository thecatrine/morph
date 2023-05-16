from PIL import Image
import logging
from torch.utils.data import Dataset, DataLoader
import hashlib
import io
import numpy as np
import random
import torch
import torchvision
import zipfile
import os
from .loader_utils import image_to_tensor, noise_img, weighted_timestep

log = logging.getLogger(__name__)

class TwitchDataset(Dataset):
    def __init__(self, z_file, namelist, max_ts=1000, alpha=0.99):
        super().__init__()
        self.max_ts = max_ts
        self.alpha = 0.99
        self.z_file = z_file
        self.namelist = namelist

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        assert idx < len(self.namelist)
        filename = self.namelist[idx]
        try:
            image_data = self.z_file.read(filename)
        except Exception as e:
            log.debug(f'Error loading image {filename}: {e}')
            return None

        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGBA')
            tensor = image_to_tensor(image)
            if tensor.shape != (3, 28, 28) and tensor.shape != (3, 112, 112):
                log.debug(f'Image {filename} has dimensions {tensor.shape}')
                return None
            steps = weighted_timestep(self.max_ts-1)+1
            img_in = noise_img(tensor, steps)
            img_out = tensor
            return (torch.tensor(steps), img_in, img_out)
        except Exception as e:
            log.debug(f'Error parsing image {filename}: {e}')
            return None

class ImagenetDataset(Dataset):
    def __init__(self, tensors, max_ts=1000, alpha=0.99):
        super().__init__()
        self.max_ts = max_ts
        self.alpha = 0.99
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        assert idx < len(self.tensors)

        try:
            tensor = self.tensors[idx]
            if tensor.shape != (3, 28, 28) and tensor.shape != (3, 112, 112):
                log.debug(f'Image has dimensions {tensor.shape}')
                return None
            steps = weighted_timestep(self.max_ts-1)+1
            img_in = noise_img(tensor, steps)
            img_out = tensor
            return (torch.tensor(steps), img_in, img_out)
        except Exception as e:
            log.debug(f'Error parsing image: {e}')
            return None

class TwitchData():
    def __init__(self, path='loaders/data/twitch_archive.zip',
                 batch_size=128, shuffle=True, num_workers=8, max_ts=1000):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.z_file = zipfile.ZipFile(path, 'r')
        self.full_namelist = self.z_file.namelist()
        self.max_ts = max_ts

    def dataloaders(self):
        random.shuffle(self.full_namelist)
        namelists = {}
        dataloaders = {}
        splits = [('val', b'\x02'), ('test', b'\x04'), ('train', None)]

        for name in self.full_namelist:
            name_hash = hashlib.sha256(name.encode('utf-8')).digest()
            for split, breakpoint in splits:
                if breakpoint is None or name_hash[:1] < breakpoint:
                    namelists.setdefault(split, []).append(name)

        default_collate = torch.utils.data.dataloader.default_collate
        def collate_fn(batch):
            batch = [x for x in batch if x is not None]
            return default_collate(batch)

        for split, namelist in namelists.items():
            dataset = TwitchDataset(self.z_file, namelist, max_ts=self.max_ts)
            dataloaders[split] = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                num_workers=self.num_workers, collate_fn=collate_fn
            )

        return dataloaders

    def __enter__(self):
        pass

    def __exit__(self):
        self.z_file.close()

class ImagenetData():
    def __init__(self, path='loaders/data/small',
                 batch_size=128, shuffle=True, num_workers=8, max_ts=1000):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_ts = max_ts
        self.path = path

    def dataloaders(self):
        namelists = {}
        dataloaders = {}

        train_tensors = torch.load(self.path+"/train_32x32.pt")
        test_tensors = torch.load(self.path+"/valid_32x32.pt")

        tensors = {
            "train": train_tensors,
            "val": test_tensors[:10000],
            "test": test_tensors[10000:],
        }

        for split, tens in tensors.items():
            dataset = ImagenetDataset(tens, max_ts=self.max_ts)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )

        return dataloaders

    def __enter__(self):
        pass

    def __exit__(self):
        pass

class BigTwitchDataset(Dataset):
    def __init__(self,
                 load_n=25,
                 path='loaders/data/twitch_big',
                 batch_size=128,
                 shuffle=True,
                 manual_shuffle=False,
                 num_workers=8,
                 max_ts=1000):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.manual_shuffle = manual_shuffle
        self.num_workers = num_workers
        self.max_ts = max_ts
        self.path = path
        self.files_to_load = [f'{path}/batch_tensors_{n}.pt' for n in range(load_n)]
        self.all_tensors = []
        for fname in self.files_to_load:
            if not os.path.exists(fname):
                continue
            ten = torch.load(fname)
            train_tensors = ten[200:]
            if self.manual_shuffle:
                train_tensors = train_tensors[torch.randperm(train_tensors.shape[0])]
            test_tensors = ten[100:200]
            val_tensors = ten[:100]
            self.all_tensors.append({
                'train': train_tensors,
                'test': test_tensors,
                'val': val_tensors,
            })
        sz=5
        self.all_tensors = [{
            'train': torch.cat([x['train'] for x in self.all_tensors[i*sz:(i+1)*sz]], dim=0),
            'test': torch.cat([x['test'] for x in self.all_tensors[i*sz:(i+1)*sz]], dim=0),
            'val': torch.cat([x['val'] for x in self.all_tensors[i*sz:(i+1)*sz]], dim=0),
        } for i in range(load_n//sz)]

    def dataloaders(self):
        if self.shuffle:
            random.shuffle(self.all_tensors)
        for tensors in self.all_tensors:
            dataloaders = {}
            for k, v in tensors.items():
                dataset = ImagenetDataset(v, max_ts=self.max_ts)
                dataloaders[k] = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                )
            yield dataloaders

class NewTwitchDataset(Dataset):
    def __init__(self, path='loaders/data/twitch',
                 batch_size=128, shuffle=True, num_workers=8, manual_shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.path = path
        self.manual_shuffle = manual_shuffle

    def dataloaders(self):
        namelists = {}
        dataloaders = {}

        train_tensors = torch.load(self.path+"/train.pt")
        val_tensors = torch.load(self.path+"/test.pt")

        tensors = {
            "train": train_tensors,
            "val": val_tensors,
        }

        for split, tens in tensors.items():
            dataset = torch.utils.data.TensorDataset(tens)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )

        yield dataloaders

    def __enter__(self):
        pass

    def __exit__(self):
        pass
