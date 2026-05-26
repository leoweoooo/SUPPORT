import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skimage.io as skio
import torch
from tqdm import tqdm

from src.dataset import DatasetSUPPORT_test_stitch
from src.model.SUPPORT import SUPPORT
from src.utils import get_device


@dataclass
class SupportTestOptions:
    data: str
    model: str
    output: str
    patch_size: list[int]
    patch_interval: list[int]
    batch_size: int
    bs_size: int
    edge_mode: str
    bp: bool


def arg_parser() -> SupportTestOptions:
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
    parser.add_argument(
        "--edge-mode",
        type=str,
        default=None,
        choices=["repeat", "mirror"],
        help="How to handle edge frames",
    )
    parser.add_argument("--bp", action="store_true")
    opt = parser.parse_args()
    return SupportTestOptions(**vars(opt))


class SupportInference:
    def __init__(self, options: SupportTestOptions) -> None:
        self.options = options
        self.device = get_device()
        print(f"Using device: {self.device}")
        self._load_metadata()
        self.model = self._build_model()

    def _load_metadata(self):
        """
        load in the training metadata from `train_metadata.json` if possible,
        and override the bs_size, bp and patch_size with the values used during training.
        """
        metadata_path = Path(self.options.model).parent / "train_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.options.bs_size = metadata["bs_size"]
            self.options.bp = metadata["bp"]
            self.options.patch_size = metadata["patch_size"]
            print(f"Loaded training metadata from {metadata_path}")
        else:
            print("Warning: no train_metadata.json found, using provided arguments.")

    def _build_model(self) -> SUPPORT:
        """
        Build and load the SUPPORT model from the provided checkpoint.

        Returns:
            SUPPORT: model loaded with trained weights, set to eval mode.
        """
        model = SUPPORT(
            in_channels=self.options.patch_size[0],
            mid_channels=[16, 32, 64, 128, 256],
            depth=5,
            blind_conv_channels=64,
            one_by_one_channels=[32, 16],
            last_layer_channels=[64, 32, 16],
            bs_size=self.options.bs_size,
            bp=self.options.bp,
        ).to(self.device)
        model.load_state_dict(torch.load(self.options.model, map_location=self.device))
        model.eval()

        return model

    def _resolve_files(self) -> tuple[list[Path], list[Path]]:
        """
        Resolve input and output file paths.

        Returns:
            Tuple of (data_files, output_files).
        """
        data_path = Path(self.options.data)
        data_files = (
            list(data_path.rglob("*.tif")) if data_path.is_dir() else [data_path]
        )

        output_path = Path(self.options.output)
        if len(data_files) > 1:
            output_path.mkdir(parents=True, exist_ok=True)
            output_files = [output_path / f"denoised_{f.name}" for f in data_files]
        else:
            output_files = [output_path]

        return data_files, output_files

    def _apply_edge_padding(self, tif: torch.Tensor) -> torch.Tensor:
        """
        Pad the temporal edges of a TIFF stack so that edge frames can be processed.

        Arguments:
            tif: input tensor with shape [T, H, W]

        Returns:
            Padded tensor.
        """
        half = self.options.patch_size[0] // 2
        if self.options.edge_mode == "repeat":
            return torch.cat(
                [
                    tif[0, :, :].unsqueeze(0).repeat((half, 1, 1)),
                    tif,
                    tif[-1, :, :].unsqueeze(0).repeat((half, 1, 1)),
                ]
            )
        elif self.options.edge_mode == "mirror":
            return torch.cat(
                [
                    tif[1 : half + 1, :, :].flip(0),
                    tif,
                    tif[-half - 1 : -1, :, :].flip(0),
                ]
            )
        return tif

    def _denoise(self, tif: torch.Tensor) -> np.ndarray:
        """
        Run inference on a single TIFF stack.

        Arguments:
            tif: input tensor with shape [T, H, W]

        Returns:
            denoised_stack: denoised numpy array with shape [T, H, W]
        """
        dataset = DatasetSUPPORT_test_stitch(
            tif,
            patch_size=self.options.patch_size,
            patch_interval=self.options.patch_interval,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.options.batch_size
        )

        denoised_stack = np.zeros(dataset.noisy_image.shape, dtype=np.float32)

        with torch.no_grad():
            for _, (noisy_image, _, single_coordinate) in enumerate(
                tqdm(dataloader, desc="denoising")
            ):
                noisy_image = noisy_image.to(self.device)
                noisy_image_denoised = self.model(noisy_image)
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

        # denormalize
        denoised_stack = (
            denoised_stack * dataset.std_image.numpy() + dataset.mean_image.numpy()
        )

        return denoised_stack

    def run(self):
        """
        Run inference on all input files and save results.
        """
        data_files, output_files = self._resolve_files()

        if self.options.edge_mode is not None:
            print(
                f'Warning: edge_mode="{self.options.edge_mode}" — edge frames will be '
                f"processed using a workaround, not an ideal solution."
            )

        for data_file, output_file in zip(data_files, output_files):
            tif = torch.from_numpy(skio.imread(str(data_file)).astype(np.float32))
            tif = self._apply_edge_padding(tif)

            denoised = self._denoise(tif)

            if self.options.edge_mode in ["repeat", "mirror"]:
                half = self.options.patch_size[0] // 2
                denoised = denoised[half:-half]

            print(f"Output: {output_file}  shape: {denoised.shape}")
            skio.imsave(str(output_file), denoised, metadata={"axes": "TYX"})


if __name__ == "__main__":
    SupportInference(arg_parser()).run()
