import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import gen_train_dataloader, random_transform
from src.model.SUPPORT import SUPPORT
from src.utils import get_device


@dataclass
class SupportTrainOptions:
    # experiment
    random_seed: int
    epoch: int
    n_epochs: int
    exp_name: str
    results_dir: str
    input_frames: int

    # dataset
    is_zarr: bool
    is_folder: bool
    noisy_data: list[str]
    patch_size: list[int]
    patch_interval: list[int]
    batch_size: int

    # model
    depth: int
    blind_conv_channels: int
    one_by_one_channels: list[int]
    last_layer_channels: list[int]
    bs_size: list[int]
    bp: bool
    unet_channels: list[int]

    # training
    lr: float
    loss_coef: list[float]

    # util
    use_amp: bool
    use_CPU: bool
    n_cpu: int
    prefetch_factor: int
    logging_interval_batch: int
    logging_interval: int
    sample_interval: int
    sample_max_t: int
    checkpoint_interval: int
    checkpoint_interval_batch: int


def arg_parser() -> SupportTrainOptions:
    """
    Parse command line arguments for SUPPORT training.

    Returns:
        SupportTrainArgs: parsed and validated arguments as a typed dataclass.
    """
    parser = argparse.ArgumentParser(description="Train the SUPPORT denoising model.")

    # experiment
    parser.add_argument(
        "--random_seed", type=int, default=0, help="random seed for rng"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="epoch to start training from (need epoch-1 model)",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=500, help="number of epochs of training"
    )
    parser.add_argument(
        "--exp_name", type=str, default="myEXP", help="name of the experiment"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="root directory to save results",
    )
    parser.add_argument(
        "--input_frames", type=int, default=61, help="number of input frames"
    )

    # dataset
    parser.add_argument("--is_zarr", action="store_true", help="noisy_data is zarr")
    parser.add_argument(
        "--is_folder", action="store_true", help="noisy_data is a folder"
    )
    parser.add_argument(
        "--noisy_data",
        type=str,
        nargs="+",
        required=True,
        help="list of paths to noisy data",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=[61, 128, 128],
        nargs="+",
        help="size of the patches",
    )
    parser.add_argument(
        "--patch_interval",
        type=int,
        default=[1, 64, 64],
        nargs="+",
        help="size of the patch interval",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of the batches"
    )

    # model
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="number of blind spot convolutions, must be odd",
    )
    parser.add_argument(
        "--blind_conv_channels",
        type=int,
        default=64,
        help="channels in blind spot convolutions",
    )
    parser.add_argument(
        "--one_by_one_channels",
        type=int,
        default=[32, 16],
        nargs="+",
        help="channels of 1x1 convolutions",
    )
    parser.add_argument(
        "--last_layer_channels",
        type=int,
        default=[64, 32, 16],
        nargs="+",
        help="channels of last layer convolutions",
    )
    parser.add_argument(
        "--bs_size", type=int, default=[3, 3], nargs="+", help="size of the blind spot"
    )
    parser.add_argument("--bp", action="store_true", help="enable blind plane mode")
    parser.add_argument(
        "--unet_channels",
        type=int,
        default=[64, 128, 256, 512, 1024],
        nargs="+",
        help="UNet channel sizes",
    )

    # training
    parser.add_argument("--lr", type=float, default=5e-4, help="adam learning rate")
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=[0.5, 0.5],
        nargs="+",
        help="L1/L2 loss coefficients",
    )

    # util
    parser.add_argument(
        "--use_amp", action="store_true", help="use automatic mixed precision"
    )
    parser.add_argument("--use_CPU", action="store_true", help="force CPU usage")
    parser.add_argument(
        "--n_cpu", type=int, default=8, help="number of CPU threads for data loading"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2, help="number of batches to prefetch"
    )
    parser.add_argument(
        "--logging_interval_batch",
        type=int,
        default=50,
        help="logging interval in batches",
    )
    parser.add_argument(
        "--logging_interval", type=int, default=1, help="logging interval in epochs"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="interval between saving samples",
    )
    parser.add_argument(
        "--sample_max_t",
        type=int,
        default=600,
        help="maximum time step for saving samples",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="checkpoint interval in epochs",
    )
    parser.add_argument(
        "--checkpoint_interval_batch",
        type=int,
        default=10000,
        help="checkpoint interval in batches",
    )

    opt = parser.parse_args()

    # validation
    if opt.input_frames != opt.patch_size[0]:
        raise ValueError("input_frames must equal patch_size[0]")
    if len(opt.loss_coef) != 2:
        raise ValueError("loss_coef must have exactly 2 values")

    # resolve file paths
    if not opt.is_zarr and opt.is_folder:
        all_files = []
        for folder in opt.noisy_data:
            all_files += sorted(
                [str(p) for p in Path(folder).rglob("*") if p.is_file()]
            )
        opt.noisy_data = all_files
    elif opt.is_zarr and opt.is_folder:
        all_dirs = []
        for folder in opt.noisy_data:
            for root, dirs, _ in os.walk(folder):
                for d in dirs:
                    if d.endswith(".zarr"):
                        all_dirs.append(os.path.join(root, d))
                dirs[:] = [d for d in dirs if not d.endswith(".zarr")]
        opt.noisy_data = sorted(all_dirs)

    print("Noisy files:")
    for f in opt.noisy_data:
        print(f"  {f}")

    return SupportTrainOptions(**vars(opt))


class SupportTrainer:
    def __init__(self, options: SupportTrainOptions):
        self.args = options
        self.device = get_device()

        if options.use_CPU:
            self.device = torch.device("cpu")

        if self.device.type == "mps" and options.use_amp:
            print("Warning: AMP is not fully supported on MPS, disabling.")
            self.args.use_amp = False

        print(f"Using device: {self.device}")

        self._setup_dirs()
        self._setup_logging()
        self.writer = SummaryWriter(
            os.path.join(options.results_dir, "tsboard", options.exp_name)
        )
        self.rng = np.random.default_rng(options.random_seed)
        self.dataloader = self._build_dataloader()
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=options.lr)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=options.use_amp)
        self._load_checkpoint()
        self._save_metadata()

    def _setup_dirs(self):
        """Create output directories for images, models, and logs."""
        args = self.args
        os.makedirs(
            os.path.join(args.results_dir, "images", args.exp_name), exist_ok=True
        )
        os.makedirs(
            os.path.join(args.results_dir, "saved_models", args.exp_name), exist_ok=True
        )
        os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)

    def _setup_logging(self):
        """Configure file-based logging."""
        args = self.args
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(args.results_dir, "logs", f"{args.exp_name}.log"),
            filemode="a",
            format="%(name)s - %(levelname)s - %(message)s",
        )

    def _build_dataloader(self):
        """Build and return the training dataloader."""
        args = self.args
        return gen_train_dataloader(
            args.patch_size,
            args.patch_interval,
            args.batch_size,
            args.noisy_data,
            args,
            is_zarr=args.is_zarr,
        )

    def _build_model(self) -> SUPPORT:
        """Build and return the SUPPORT model."""
        args = self.args
        return SUPPORT(
            in_channels=args.input_frames,
            mid_channels=args.unet_channels,
            depth=args.depth,
            blind_conv_channels=args.blind_conv_channels,
            one_by_one_channels=args.one_by_one_channels,
            last_layer_channels=args.last_layer_channels,
            bs_size=args.bs_size,
            bp=args.bp,
        ).to(self.device)

    def _load_checkpoint(self):
        """Resume training from a saved checkpoint if epoch > 0."""
        args = self.args
        if args.epoch == 0:
            return
        model_path = os.path.join(
            args.results_dir,
            "saved_models",
            args.exp_name,
            f"model_{args.epoch - 1}.pth",
        )
        optimizer_path = os.path.join(
            args.results_dir,
            "saved_models",
            args.exp_name,
            f"optimizer_{args.epoch - 1}.pth",
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=self.device)
        )
        if args.use_amp:
            scaler_path = os.path.join(
                args.results_dir,
                "saved_models",
                args.exp_name,
                f"scaler_{args.epoch - 1}.pth",
            )
            self.scaler.load_state_dict(
                torch.load(scaler_path, map_location=self.device)
            )
        print(f"Resumed from epoch {args.epoch - 1}")

    def _save_metadata(self):
        """Save training metadata to JSON for use during inference."""
        args = self.args
        metadata = {
            "bs_size": args.bs_size,
            "bp": args.bp,
            "input_frames": args.input_frames,
            "unet_channels": args.unet_channels,
            "depth": args.depth,
            "blind_conv_channels": args.blind_conv_channels,
            "one_by_one_channels": args.one_by_one_channels,
            "last_layer_channels": args.last_layer_channels,
            "patch_size": args.patch_size,
            "patch_interval": args.patch_interval,
        }
        metadata_path = os.path.join(
            args.results_dir, "saved_models", args.exp_name, "train_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _train_epoch(self, epoch: int) -> tuple[list, list, list]:
        """
        Train the model for a single epoch.

        Arguments:
            epoch: current epoch index

        Returns:
            Tuple of (loss_list, loss_list_l1, loss_list_l2)
        """
        args = self.args
        is_rotate = self.model.bs_size[0] == self.model.bs_size[1]

        self.model.train()
        loss_list, loss_list_l1, loss_list_l2 = [], [], []

        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()

        for i, data in enumerate(tqdm(self.dataloader)):
            if args.is_zarr:
                noisy_image, _, ds_idx, noisy_image_avg, noisy_image_std = data
                noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1)).to(
                    self.device
                )
                noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1)).to(
                    self.device
                )
            else:
                noisy_image, _, ds_idx = data

            _, T, _, _ = noisy_image.shape
            noisy_image = noisy_image.to(self.device)
            noisy_image, _ = random_transform(noisy_image, None, self.rng, is_rotate)

            if args.is_zarr:
                noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std

            noisy_image_target = torch.unsqueeze(noisy_image[:, T // 2, :, :], dim=1)

            self.optimizer.zero_grad()
            with torch.amp.autocast(self.device.type, enabled=args.use_amp):
                noisy_image_denoised = self.model(noisy_image)
                loss_l1 = L1_pixelwise(noisy_image_denoised, noisy_image_target)
                loss_l2 = L2_pixelwise(noisy_image_denoised, noisy_image_target)
                loss_sum = args.loss_coef[0] * loss_l1 + args.loss_coef[1] * loss_l2

            self.scaler.scale(loss_sum).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_list_l1.append(loss_l1.item())
            loss_list_l2.append(loss_l2.item())
            loss_list.append(loss_sum.item())

            if (epoch % args.logging_interval == 0) and (
                i % args.logging_interval_batch == 0
            ):
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                loss_mean = np.mean(loss_list)
                loss_mean_l1 = np.mean(loss_list_l1)
                loss_mean_l2 = np.mean(loss_list_l2)
                self.writer.add_scalar(
                    "Loss_l1/train_batch",
                    loss_mean_l1,
                    epoch * len(self.dataloader) + i,
                )
                self.writer.add_scalar(
                    "Loss_l2/train_batch",
                    loss_mean_l2,
                    epoch * len(self.dataloader) + i,
                )
                self.writer.add_scalar(
                    "Loss/train_batch", loss_mean, epoch * len(self.dataloader) + i
                )
                logging.info(
                    f"[{ts}] Epoch [{epoch}/{args.n_epochs}] Batch [{i + 1}/{len(self.dataloader)}] "
                    f"loss: {loss_mean:.4f}, l1: {loss_mean_l1:.4f}, l2: {loss_mean_l2:.4f}"
                )

            if (args.checkpoint_interval != -1) and (
                i % args.checkpoint_interval_batch == 0
            ):
                self._save_checkpoint(epoch, batch=i)

        return loss_list, loss_list_l1, loss_list_l2

    def _save_checkpoint(self, epoch: int, batch: int | None = None):
        """Save model, optimizer, and scaler state to disk."""
        args = self.args
        suffix = f"_{epoch}_batch_{batch}" if batch is not None else f"_{epoch}"
        base = os.path.join(args.results_dir, "saved_models", args.exp_name)
        torch.save(self.model.state_dict(), os.path.join(base, f"model{suffix}.pth"))
        torch.save(
            self.optimizer.state_dict(), os.path.join(base, f"optimizer{suffix}.pth")
        )
        if args.use_amp:
            torch.save(
                self.scaler.state_dict(), os.path.join(base, f"scaler{suffix}.pth")
            )

    def run(self):
        """Run the full training loop."""
        args = self.args
        for epoch in range(args.epoch, args.n_epochs):
            self.dataloader.dataset.precompute_indices()
            loss_list, loss_list_l1, loss_list_l2 = self._train_epoch(epoch)

            if epoch % args.logging_interval == 0:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                loss_mean = np.mean(loss_list)
                loss_mean_l1 = np.mean(loss_list_l1)
                loss_mean_l2 = np.mean(loss_list_l2)
                self.writer.add_scalar("Loss/train", loss_mean, epoch)
                self.writer.add_scalar("Loss_l1/train", loss_mean_l1, epoch)
                self.writer.add_scalar("Loss_l2/train", loss_mean_l2, epoch)
                logging.info(
                    f"[{ts}] Epoch [{epoch}/{args.n_epochs}] "
                    f"loss: {loss_mean:.4f}, l1: {loss_mean_l1:.4f}, l2: {loss_mean_l2:.4f}"
                )

            if (args.checkpoint_interval != -1) and (
                epoch % args.checkpoint_interval == 0
            ):
                self._save_checkpoint(epoch)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    SupportTrainer(arg_parser()).run()
