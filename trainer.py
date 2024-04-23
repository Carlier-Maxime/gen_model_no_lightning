import os
import tempfile
from typing import List

import torch
import torch.types
import torch.distributed
from tqdm import trange, tqdm
from torch import optim

from callback import Callback


def subprocess_fn(rank: int, trainer, local_rank: int = -1, temp_dir=None, args=None):
    trainer.global_rank = rank
    trainer.device = torch.device(trainer.accelerator, rank if local_rank == -1 else local_rank)
    trainer.init_distributed(temp_dir)
    trainer.fit(**args, launch_multiprocessing=False)


class Trainer:
    def __init__(self, devices: int | str | torch.types.Device, accelerator: str, max_epochs: int, callbacks: List[Callback], **kwargs):
        self.devices = devices
        self.accelerator = 'cuda' if accelerator == 'gpu' else accelerator
        self.max_epochs = max_epochs
        self.global_rank = 0
        self.model = None
        self.data = None
        self.ckpt_path = None
        self.callbacks = callbacks
        self.num_gpus = 0
        self.device = None
        self.global_step = 0

    def save_checkpoint(self, ckpt_path: str | None = None) -> None:
        assert self.model is not None
        if ckpt_path is None:
            assert self.ckpt_path is not None
        else:
            self.ckpt_path = ckpt_path
        torch.save(self.model.state_dict(), ckpt_path)

    def fit(self, model: torch.nn.Module | None, data, ckpt_path: str | None = None, launch_multiprocessing: bool = True):
        if model is None:
            assert self.model is not None
        else:
            self.model = model
        if ckpt_path is not None: self.ckpt_path = ckpt_path
        if launch_multiprocessing:
            self.launch_multiprocessing(model=model, data=data, ckpt_path=self.ckpt_path)
            return
        for callback in self.callbacks: callback.on_fit_start(self, model)
        optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)
        self.global_step = 0
        for _ in trange(self.max_epochs, unit='epoch', leave=True):
            batch_idx = 0
            for batch in tqdm(data.train_dataset, unit='batch', leave=True):
                for callback in self.callbacks: callback.on_train_batch_start(self, model, batch, batch_idx)
                batch_idx += 1


    def launch_multiprocessing(self, use_idr_torch: bool = False, **args):
        if use_idr_torch:
            import idr_torch
            self.devices = None
            self.num_gpus = idr_torch.size
            subprocess_fn(rank=idr_torch.rank, trainer=self, local_rank=idr_torch.local_rank, temp_dir=None)
        else:
            if isinstance(self.devices, torch.device) or isinstance(self.devices, int):
                self.num_gpus = 1
            elif isinstance(self.devices, str):
                self.devices = self.devices.split(',')
                self.num_gpus = len(self.devices)
            torch.multiprocessing.set_start_method('spawn')
            with tempfile.TemporaryDirectory() as temp_dir:
                if self.num_gpus == 1:
                    subprocess_fn(rank=0, trainer=self, temp_dir=temp_dir)
                else:
                    local_rank = -1
                    torch.multiprocessing.spawn(fn=subprocess_fn, args=(self, local_rank, temp_dir, args), nprocs=self.num_gpus)

    def init_distributed(self, temp_dir):
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init')) if temp_dir is not None else None
        if os.name == 'nt':
            init_method = ('file:///' + init_file.replace('\\', '/')) if init_file is not None else 'env:///'
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=self.global_rank, world_size=self.num_gpus)
        else:
            init_method = f'file://{init_file}' if init_file is not None else 'env://'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=self.global_rank, world_size=self.num_gpus)
        torch.distributed.barrier()
