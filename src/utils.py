import torch


def get_device() -> torch.device:
    """
    returns the best available pytorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
