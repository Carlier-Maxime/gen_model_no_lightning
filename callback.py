import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import torch

from sgm.util import isheatmap
from omegaconf import OmegaConf


class Trainer:
    pass


class Callback(object):
    def __init__(self):
        pass

    def on_fit_start(self, trainer, model):
        pass

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        pass

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        pass


class ImageLogger(Callback):
    def __init__(
            self,
            batch_frequency,
            max_images,
            clamp=True,
            increase_log_steps=True,
            rescale=True,
            disabled=False,
            log_on_batch_idx=False,
            log_first_step=False,
            log_images_kwargs=None,
            log_before_first_step=False,
            enable_autocast=True,
            log_input: bool = True,
            log_reconstruction: bool = True,
            log_samples: bool = True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled or self.max_images <= 0
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step
        self.log_input = log_input
        self.log_reconstruction = log_reconstruction
        self.log_samples = log_samples

    # @rank_zero_only
    def log_local(
            self,
            save_dir,
            split,
            images,
            global_step,
            current_epoch,
            batch_idx
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if k == "inputs" and not self.log_input: continue
            if k == "reconstructions" and not self.log_reconstruction: continue
            if k == "samples" and not self.log_samples: continue
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)

    # @rank_zero_only
    def log_img(self, trainer, model, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else trainer.global_step
        if (
                self.check_frequency(check_idx)
                and hasattr(model, "log_images")  # batch_idx % self.batch_freq == 0
                and callable(model.log_images)
        ):
            is_train = model.training
            if is_train:
                model.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = model.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()
                    if self.clamp and not isheatmap(images[k]):
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                trainer.logdir,
                split,
                images,
                trainer.global_step,
                trainer.current_epoch,
                batch_idx
            )

            if is_train:
                model.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if trainer.global_rank != 0: return
        if not self.disabled and (trainer.global_step > 0 or self.log_first_step):
            self.log_img(trainer, model, batch, batch_idx, split="train")

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if trainer.global_rank != 0: return
        if self.log_before_first_step and trainer.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(trainer, model, batch, batch_idx, split="train")


class SetupCallback(Callback):
    def __init__(
            self,
            resume,
            now,
            logdir,
            ckptdir,
            cfgdir,
            config,
            debug,
            ckpt_name=None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: Trainer, pl_module, exception):
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt" if self.ckpt_name is None else self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank != 0: return
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)
        if "callbacks" in self.config:
            if "metrics_over_trainsteps_checkpoint" in self.config["callbacks"]:
                os.makedirs(os.path.join(self.ckptdir, "trainstep_checkpoints"), exist_ok=True)
        print("Project config")
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(
            self.config,
            os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
        )


class CheckpointLogger(Callback):
    def __init__(self, ckptdir: str, batch_frequency: int, log_first_step: bool = False):
        super().__init__()
        self.ckptdir = ckptdir
        self.batch_frequency = batch_frequency
        self.log_first_step = log_first_step

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if (trainer.global_rank != 0) or (trainer.global_step == 0 and not self.log_first_step) or (trainer.global_step % self.batch_frequency != 0): return
        trainer.save_checkpoint(os.path.join(self.ckptdir, "weights_gs-{:06}_e-{:06}_b-{:06}.ckpt".format(trainer.global_step, trainer.current_epoch, batch_idx)))
