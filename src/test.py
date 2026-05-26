import argparse
from dataclasses import dataclass

import numpy as np
import skimage.io as skio
import torch
from tqdm import tqdm

from model.SUPPORT import SUPPORT
from src.utils.dataset import DatasetSUPPORT_test_stitch


def validate(test_dataloader, model):
    """
    Validate a model with a test data

    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetFRECTAL_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(
            test_dataloader.dataset.noisy_image.shape, dtype=np.float32
        )

        # stitching denoised stack
        # insert the results if the stack value was NaN
        # or, half of the output volume
        for _, (noisy_image, _, single_coordinate) in enumerate(
            tqdm(test_dataloader, desc="validate")
        ):
            noisy_image = noisy_image.cuda()  # [b, z, y, x]
            noisy_image_denoised = model(noisy_image)
            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)):
                stack_start_w = int(single_coordinate["stack_start_w"][bi])
                stack_end_w = int(single_coordinate["stack_end_w"][bi])
                patch_start_w = int(single_coordinate["patch_start_w"][bi])
                patch_end_w = int(single_coordinate["patch_end_w"][bi])

                stack_start_h = int(single_coordinate["stack_start_h"][bi])
                stack_end_h = int(single_coordinate["stack_end_h"][bi])
                patch_start_h = int(single_coordinate["patch_start_h"][bi])
                patch_end_h = int(single_coordinate["patch_end_h"][bi])

                stack_start_s = int(single_coordinate["init_s"][bi])

                denoised_stack[
                    stack_start_s + (T // 2),
                    stack_start_h:stack_end_h,
                    stack_start_w:stack_end_w,
                ] = (
                    noisy_image_denoised[bi]
                    .squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w]
                    .cpu()
                )

        # change nan values to 0 and denormalize
        denoised_stack = (
            denoised_stack * test_dataloader.dataset.std_image.numpy()
            + test_dataloader.dataset.mean_image.numpy()
        )

        return denoised_stack


@dataclass
class SupportTestArgs:
    data: str
    model: str
    output: str
    patch_size: list[int]
    patch_interval: list[int]
    batch_size: int
    bs_size: int
    bp: bool


def arg_parser() -> SupportTestArgs:
    """
    Parse command line arguments for SUPPORT inference.

    Returns:
        SupportTestArgs: parsed arguments as a typed dataclass.
    """
    parser = argparse.ArgumentParser(description="Run SUPPORT denoising interface.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--patch-size", type=int, nargs=3, default=[61, 64, 64], metavar=("T", "X", "Y")
    )
    parser.add_argument(
        "--patch-interval",
        type=int,
        nargs=3,
        default=[1, 32, 32],
        metavar=("T", "X", "Y"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bs-size", type=int, default=3)
    parser.add_argument("--bp", action="store_true")
    ns = parser.parse_args()
    return SupportTestArgs(**vars(ns))


if __name__ == "__main__":
    args = arg_parser()
    model = SUPPORT(
        in_channels=args.patch_size[0],
        mid_channels=[16, 32, 64, 128, 256],
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=[64, 32, 16],
        bs_size=args.bs_size,
        bp=args.bp,
    ).cuda()

    model.load_state_dict(torch.load(args.model))

    demo_tif = torch.from_numpy(skio.imread(args.data).astype(np.float32)).type(
        torch.FloatTensor
    )

    testset = DatasetSUPPORT_test_stitch(
        demo_tif, patch_size=args.patch_size, patch_interval=args.patch_interval
    )

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)
    denoised_stack = validate(testloader, model)

    print(denoised_stack.shape)
    skio.imsave(
        args.output,
        denoised_stack[
            (model.in_channels - 1) // 2 : -(model.in_channels - 1) // 2, :, :
        ],
        metadata={"axes": "TYX"},
    )
