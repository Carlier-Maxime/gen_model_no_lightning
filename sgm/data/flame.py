import os
import zipfile

import PIL.Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class FLAMEDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._zipfile = None
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

    def __getitem__(self, i):
        img = torch.from_numpy(np.array(PIL.Image.open(os.path.join(self._path, self._image_fnames[i])).convert('RGB'))).permute(2, 0, 1).divide(255).mul(2).sub(1)
        return {"jpg": img, "cls": img.mean()}

    def __len__(self):
        return len(self._image_fnames)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile


class FLAMELoader(torch.nn.Module):
    def __init__(self, path, batch_size, num_workers=0, prefetch_factor=2, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else 0
        self.shuffle = shuffle
        self.train_dataset = FLAMEDataset(path)
        self.test_dataset = FLAMEDataset(path)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
