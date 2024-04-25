import os
import tempfile
from typing import List

import torch
import torch.types
import torch.distributed
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from torch import optim

from callback import Callback


def subprocess_fn(rank: int, trainer, local_rank: int = -1, temp_dir=None, args=None):
    trainer.global_rank = rank
    if local_rank == -1: local_rank = rank
    trainer.device = torch.device(trainer.accelerator, local_rank if trainer.devices is None else int(trainer.devices[local_rank]))
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

    def fit(self, model: torch.nn.Module | None, data, ckpt_path: str | None = None, launch_multiprocessing: bool = True, use_idr_torch: bool = False):
        if model is None:
            assert self.model is not None
        else:
            self.model = model
        if ckpt_path is not None: self.ckpt_path = ckpt_path
        if launch_multiprocessing:
            self.launch_multiprocessing(model=model, data=data, ckpt_path=self.ckpt_path, use_idr_torch=use_idr_torch)
            return
        for callback in self.callbacks: callback.on_fit_start(self, model)
        model = model.to(self.device)
        optimizer = model.configure_optimizers()
        self.global_step = 0
        sampler = torch.utils.data.distributed.DistributedSampler(data.train_dataset, num_replicas=self.num_gpus, rank=self.global_rank, shuffle=True)
        dataloader = DataLoader(data.train_dataset, sampler=sampler, batch_size=data.batch_size // self.num_gpus)
        for param in list(model.parameters()) + list(model.buffers()):
            if param.numel() > 0 and self.num_gpus > 1:
                torch.distributed.broadcast(param, src=0)
        for self.current_epoch in trange(self.max_epochs, unit='epoch', leave=True, disable=self.global_rank != 0):
            batch_idx = 0
            p_bar = tqdm(dataloader, unit='batch', leave=True, disable=self.global_rank != 0)
            p_bar.set_postfix(loss='?')
            for batch in p_bar:
                for key, value in batch.items(): batch[key] = value.to(self.device)
                for callback in self.callbacks: callback.on_train_batch_start(self, model, batch, batch_idx)
                outs = model(batch)
                loss = outs[0]
                p_bar.set_postfix(loss=loss.item())
                loss.backward()

                params = [param for param in model.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if self.num_gpus > 1:
                        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
                        flat /= self.num_gpus
                    torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)

                optimizer.step()
                optimizer.zero_grad()
                for callback in self.callbacks: callback.on_train_batch_end(self, model, outs[1], batch, batch_idx)
                batch_idx += 1
                self.global_step += 1

    def launch_multiprocessing(self, use_idr_torch: bool = False, **args):
        if use_idr_torch:
            import idr_torch
            self.devices = None
            self.num_gpus = idr_torch.size
            subprocess_fn(rank=idr_torch.rank, trainer=self, local_rank=idr_torch.local_rank, temp_dir=None, args=args)
        else:
            if isinstance(self.devices, torch.device) or isinstance(self.devices, int):
                self.num_gpus = 1
            elif isinstance(self.devices, str):
                self.devices = self.devices.split(',')
                self.num_gpus = len(self.devices)
                if len(self.devices[-1]) == 0: self.num_gpus -= 1
            torch.multiprocessing.set_start_method('spawn')
            with tempfile.TemporaryDirectory() as temp_dir:
                if self.num_gpus == 1:
                    subprocess_fn(rank=0, trainer=self, temp_dir=temp_dir, args=args)
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
