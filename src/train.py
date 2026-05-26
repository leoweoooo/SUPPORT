import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.SUPPORT import SUPPORT
from src.utils.dataset import gen_train_dataloader, random_transform


def train(train_dataloader, model, optimizer, scaler, rng, writer, epoch, opt):
    """
    Train a model for a single epoch

    Arguments:
        train_dataloader: (Pytorch DataLoader)
        model: (Pytorch nn.Module)
        optimizer: (Pytorch optimzer)
        rng: numpy random number generator
        writer: (Tensorboard writer)
        epoch: epoch of training (int)
        opt: argparse dictionary

    Returns:
        loss_list: list of total loss of each batch ([float])
        loss_list_l1: list of L1 loss of each batch ([float])
        loss_list_l2: list of L2 loss of each batch ([float])
        corr_list: list of correlation of each batch ([float])
    """

    is_rotate = True if model.bs_size[0] == model.bs_size[1] else False

    # initialize
    model.train()
    loss_list_l1 = []
    loss_list_l2 = []
    loss_list = []

    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()
    loss_coef = opt.loss_coef

    # training
    for i, data in enumerate(tqdm(train_dataloader)):
        if opt.is_zarr:
            (noisy_image, _, ds_idx, noisy_image_avg, noisy_image_std) = data
            noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1))
            noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1))
        else:
            (noisy_image, _, ds_idx) = data

        B, T, X, Y = noisy_image.shape
        noisy_image = noisy_image.cuda()
        noisy_image, _ = random_transform(noisy_image, None, rng, is_rotate)
        if opt.is_zarr:
            noisy_image_avg = noisy_image_avg.cuda()
            noisy_image_std = noisy_image_std.cuda()
            noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std
        noisy_image_target = torch.unsqueeze(noisy_image[:, int(T / 2), :, :], dim=1)

        optimizer.zero_grad()
        # Forward pass wrapped in autocast for AMP
        with torch.cuda.amp.autocast(enabled=opt.use_amp):
            noisy_image_denoised = model(noisy_image)
            loss_l1_pixelwise = L1_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_l2_pixelwise = L2_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_sum = (
                loss_coef[0] * loss_l1_pixelwise + loss_coef[1] * loss_l2_pixelwise
            )

        # Backward pass with GradScaler if AMP is enabled
        scaler.scale(loss_sum).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_list_l1.append(loss_l1_pixelwise.item())
        loss_list_l2.append(loss_l2_pixelwise.item())
        loss_list.append(loss_sum.item())

        # print log
        if (epoch % opt.logging_interval == 0) and (
            i % opt.logging_interval_batch == 0
        ):
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            writer.add_scalar(
                "Loss_l1/train_batch", loss_mean_l1, epoch * len(train_dataloader) + i
            )
            writer.add_scalar(
                "Loss_l2/train_batch", loss_mean_l2, epoch * len(train_dataloader) + i
            )
            writer.add_scalar(
                "Loss/train_batch", loss_mean, epoch * len(train_dataloader) + i
            )

            logging.info(
                f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] Batch [{i + 1}/{len(train_dataloader)}] "
                + f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}"
            )

        # save model, optimizer, and scaler
        if (opt.checkpoint_interval != -1) and (i % opt.checkpoint_interval_batch == 0):
            torch.save(
                model.state_dict(),
                opt.results_dir
                + "/saved_models/%s/model_%d_batch_%d.pth" % (opt.exp_name, epoch, i),
            )
            torch.save(
                optimizer.state_dict(),
                opt.results_dir
                + "/saved_models/%s/optimizer_%d_batch_%d.pth"
                % (opt.exp_name, epoch, i),
            )
            if opt.use_amp:
                torch.save(
                    scaler.state_dict(),
                    opt.results_dir
                    + "/saved_models/%s/scaler_%d_batch_%d.pth"
                    % (opt.exp_name, epoch, i),
                )

    return loss_list, loss_list_l1, loss_list_l2


@dataclass
class SupportTrainArgs:
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


def arg_parser() -> SupportTrainArgs:
    parser = argparse.ArgumentParser()
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
        "--input_frames", type=int, default=61, help="# of input frames"
    )
    # parser.add_argument("--cuda_device", type=int, default=[0], nargs="+", help="cuda devices to use")

    # dataset
    parser.add_argument("--is_zarr", action="store_true", help="noisy_data is zarr")
    parser.add_argument("--is_folder", action="store_true", help="noisy_data is folder")
    parser.add_argument(
        "--noisy_data", type=str, nargs="+", help="List of path to the noisy data"
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
        help="the number of blind spot convolutions, must be an odd number",
    )
    parser.add_argument(
        "--blind_conv_channels",
        type=int,
        default=64,
        help="the number of channels of blind spot convolutions",
    )
    parser.add_argument(
        "--one_by_one_channels",
        type=int,
        default=[32, 16],
        nargs="+",
        help="the number of channels of 1x1 convolutions",
    )
    parser.add_argument(
        "--last_layer_channels",
        type=int,
        default=[64, 32, 16],
        nargs="+",
        help="the number of channels of 1x1 convs after UNet",
    )
    parser.add_argument(
        "--bs_size",
        type=int,
        default=[3, 3],
        nargs="+",
        help="the size of the blind spot",
    )
    parser.add_argument("--bp", action="store_true", help="blind plane")
    parser.add_argument(
        "--unet_channels",
        type=int,
        default=[64, 128, 256, 512, 1024],
        nargs="+",
        help="the number of channels of UNet",
    )

    # training
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=[0.5, 0.5],
        nargs="+",
        help="L1/L2 loss coefficients",
    )

    # util
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision for training",
    )
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2, help="number of batches to prefetch"
    )
    parser.add_argument(
        "--logging_interval_batch",
        type=int,
        default=50,
        help="interval between logging info (in batches)",
    )
    parser.add_argument(
        "--logging_interval",
        type=int,
        default=1,
        help="interval between logging info (in epochs)",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="interval between saving denoised samples",
    )
    parser.add_argument(
        "--sample_max_t",
        type=int,
        default=600,
        help="maximum time step of saving sample",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="interval between saving trained models (in epochs)",
    )
    parser.add_argument(
        "--checkpoint_interval_batch",
        type=int,
        default=10000,
        help="interval between saving trained models (in batches)",
    )
    opt = parser.parse_args()

    # argument checking
    if (opt.input_frames) != opt.patch_size[0]:
        raise Exception("input frames must be equal to z-frames of patch_size")
    if len(opt.loss_coef) != 2:
        raise Exception("loss_coef must be length-2 array")

    if not opt.is_zarr:
        if opt.is_folder:
            all_files = []

            for i in opt.noisy_data:
                all_files += sorted([str(p) for p in Path(i).rglob("*") if p.is_file()])

            opt.noisy_data = all_files
    else:
        if opt.is_folder:
            all_dirs = []
            for folder in opt.noisy_data:
                for root, dirs, files in os.walk(folder):
                    for d in dirs:
                        if d.endswith(".zarr"):
                            all_dirs.append(os.path.join(root, d))
                    dirs[:] = [d for d in dirs if not d.endswith(".zarr")]
            opt.noisy_data = sorted(all_dirs)

    # print the noisy files
    print("Noisy files:")
    for i in opt.noisy_data:
        print(i)

    return SupportTrainArgs(**vars(opt))


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    opt = arg_parser()
    cuda = torch.cuda.is_available() and (not opt.use_CPU)
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    rng = np.random.default_rng(opt.random_seed)

    os.makedirs(opt.results_dir + "/images/{}".format(opt.exp_name), exist_ok=True)
    os.makedirs(
        opt.results_dir + "/saved_models/{}".format(opt.exp_name), exist_ok=True
    )
    os.makedirs(opt.results_dir + "/logs".format(opt.exp_name), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=opt.results_dir + "/logs/{}.log".format(opt.exp_name),
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
    )
    writer = SummaryWriter(opt.results_dir + "/tsboard/{}".format(opt.exp_name))

    # -----------
    # Dataset
    # ----------
    dataloader_train = gen_train_dataloader(
        opt.patch_size,
        opt.patch_interval,
        opt.batch_size,
        opt.noisy_data,
        opt,
        is_zarr=opt.is_zarr,
    )

    # ----------
    # Model, Optimizers, and Loss
    # ----------
    model = SUPPORT(
        in_channels=opt.input_frames,
        mid_channels=opt.unet_channels,
        depth=opt.depth,
        blind_conv_channels=opt.blind_conv_channels,
        one_by_one_channels=opt.one_by_one_channels,
        last_layer_channels=opt.last_layer_channels,
        bs_size=opt.bs_size,
        bp=opt.bp,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # Initialize GradScaler if AMP is enabled
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    if cuda:
        model = model.cuda()

    if opt.epoch != 0:
        model.load_state_dict(
            torch.load(
                opt.results_dir
                + "/saved_models/%s/model_%d.pth" % (opt.exp_name, opt.epoch - 1)
            )
        )
        optimizer.load_state_dict(
            torch.load(
                opt.results_dir
                + "/saved_models/%s/optimizer_%d.pth" % (opt.exp_name, opt.epoch - 1)
            )
        )
        if opt.use_amp:
            scaler.load_state_dict(
                torch.load(
                    opt.results_dir
                    + "/saved_models/%s/scaler_%d.pth" % (opt.exp_name, opt.epoch - 1)
                )
            )
        print(
            "Loaded pre-trained model and optimizer weights of epoch {}".format(
                opt.epoch - 1
            )
        )

    # store metadata (to use in test.py later on)
    train_metadata = {
        "bs_size": opt.bs_size,
        "bp": opt.bp,
        "input_frames": opt.input_frames,
        "unet_channels": opt.unet_channels,
        "depth": opt.depth,
        "blind_conv_channels": opt.blind_conv_channels,
        "one_by_one_channels": opt.one_by_one_channels,
        "last_layer_channels": opt.last_layer_channels,
        "patch_size": opt.patch_size,
        "patch_interval": opt.patch_interval,
    }

    metadata_path = os.path.join(
        opt.results_dir, "saved_models", opt.exp_name, "train_metadata.json"
    )

    with open(metadata_path, "w") as f:
        json.dump(train_metadata, f, indent=2)

    # ----------
    # Training & Validation
    # ----------
    for epoch in range(opt.epoch, opt.n_epochs):
        dataloader_train.dataset.precompute_indices()
        loss_list, loss_list_l1, loss_list_l2 = train(
            dataloader_train, model, optimizer, scaler, rng, writer, epoch, opt
        )

        # logging
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if epoch % opt.logging_interval == 0:
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))

            writer.add_scalar("Loss/train", loss_mean, epoch)
            writer.add_scalar("Loss_l1/train", loss_mean_l1, epoch)
            writer.add_scalar("Loss_l2/train", loss_mean_l2, epoch)
            logging.info(
                f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] "
                + f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}"
            )

        if (opt.checkpoint_interval != -1) and (epoch % opt.checkpoint_interval == 0):
            torch.save(
                model.state_dict(),
                opt.results_dir
                + "/saved_models/%s/model_%d.pth" % (opt.exp_name, epoch),
            )
            torch.save(
                optimizer.state_dict(),
                opt.results_dir
                + "/saved_models/%s/optimizer_%d.pth" % (opt.exp_name, epoch),
            )
            if opt.use_amp:
                torch.save(
                    scaler.state_dict(),
                    opt.results_dir
                    + "/saved_models/%s/scaler_%d.pth" % (opt.exp_name, epoch),
                )

        # if (epoch % opt.sample_interval == 0):
        #     skio.imsave(opt.results_dir + "/images/%s/denoised_%d.pth" % (opt.exp_name, epoch), )
