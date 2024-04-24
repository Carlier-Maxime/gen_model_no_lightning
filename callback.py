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
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

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
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
                self.check_frequency(check_idx)
                and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
                and callable(pl_module.log_images)
                and
                # batch_idx > 5 and
                self.max_images > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = pl_module.log_images(
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
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx
            )

            if is_train:
                pl_module.train()

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
            self.log_img(model, batch, batch_idx, split="train")

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if trainer.global_rank != 0: return
        if self.log_before_first_step and trainer.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(model, batch, batch_idx, split="train")

    # @rank_zero_only
    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                    pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


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
            if self.ckpt_name is None:
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            else:
                ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.config:
                if (
                        "metrics_over_trainsteps_checkpoint"
                        in self.config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))

            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
