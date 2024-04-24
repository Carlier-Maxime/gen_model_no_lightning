import argparse
import datetime
import glob
import os
import sys

import numpy as np
import torch
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version

from sgm.util import instantiate_from_config
from trainer import Trainer


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--no_date",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--projectname",
        type=str,
        default="stablediffusion",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--legacy_naming",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="name run based on config file name if true, else by whole path",
    )
    parser.add_argument(
        "--enable_tf32",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12",
    )
    parser.add_argument(
        "--startup",
        type=str,
        default=None,
        help="Startuptime from distributed script",
    )
    parser.add_argument(
        "--use_slurm",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use slurm for devices and nb nodes",
    )
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from",
        )
    return parser


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print('available "last" checkpoints:')
    print(ckpt)
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print("version confusion but not bad")
            print(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print("#" * 100)
        print(f'Resuming from checkpoint "{ckpt}"')
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            if opt.legacy_naming:
                cfg_fname = os.path.split(opt.base[0])[-1]
                cfg_name = os.path.splitext(cfg_fname)[0]
            else:
                assert "configs" in os.path.split(opt.base[0])[0], os.path.split(
                    opt.base[0]
                )[0]
                cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                           os.path.split(opt.base[0])[0].split(os.sep).index("configs")
                           + 1:
                           ]  # cut away the first one (we assert all configs are in "configs")
                cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0]
                cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
            name = "_" + cfg_name
        else:
            name = ""
        if not opt.no_date:
            nowname = now + name + opt.postfix
        else:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        logdir = os.path.join(opt.logdir, nowname)
        print(f"LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        # merge trainer cli with config
        trainer_config = config.get("trainer", OmegaConf.create())
        trainer_config["accelerator"] = "gpu"

        ckpt_resume_path = opt.resume_from_checkpoint

        gpuinfo = trainer_config["devices"]
        print(f"Running on GPUs {gpuinfo}")
        trainer_opt = argparse.Namespace(**trainer_config)
        config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "callback.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "debug": opt.debug,
                    "ckpt_name": melk_ckpt_name,
                },
            },
            "image_logger": {
                "target": "callback.ImageLogger",
                "params": {"batch_frequency": 1000, "max_images": 4, "clamp": True},
            }
        }

        callbacks_cfg = config.callbacks if "callbacks" in config else OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # cmd line trainer args (which are in trainer_opt) have always priority over config-trainer-args (which are in trainer_kwargs)
        trainer_opt = vars(trainer_opt)
        trainer_kwargs = {
            key: val for key, val in trainer_kwargs.items() if key not in trainer_opt
        }
        if opt.use_slurm:
            trainer_opt['devices'] = int(os.environ['SLURM_GPUS_ON_NODE'])
            trainer_opt['num_nodes'] = int(os.environ['SLURM_NNODES'])
        trainer = Trainer(**trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        # data
        data = instantiate_from_config(config.data)
        data.prepare_data()
        # data.setup()
        print("#### Data #####")
        try:
            for k in data.datasets:
                print(
                    f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
                )
        except:
            print("datasets not yet initialized.")

        # configure learning rate
        if "batch_size" in config.data.params:
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        else:
            bs, base_lr = (
                config.data.params.train.loader.batch_size,
                config.model.base_learning_rate,
            )
        ngpu = len(config.trainer.devices.strip(",").split(","))
        if "accumulate_grad_batches" in config.trainer:
            accumulate_grad_batches = config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        def melk():
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                if melk_ckpt_name is None:
                    ckpt_path = os.path.join(ckptdir, "last.ckpt")
                else:
                    ckpt_path = os.path.join(ckptdir, melk_ckpt_name)
                trainer.save_checkpoint(ckpt_path)


        # run
        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=ckpt_resume_path)
            except Exception:
                if not opt.debug:
                    melk()
                raise
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
