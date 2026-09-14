import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_arg: str):
    if device_arg == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available."
            )
        return torch.device("cuda")

    raise ValueError(
        f"Unknown device: {device_arg}"
    )


def make_output_dir(output_dir: str):
    path = Path(output_dir)
    path.mkdir(
        parents=True,
        exist_ok=True
    )
    return path

if __name__ == "__main__":
    set_seed(0)

    device = get_device("auto")
    output_dir = make_output_dir(
        "outputs/test_run"
    )

    print("device =", device)
    print("output_dir =", output_dir)

    print("numpy random =", np.random.rand())
    print("torch random =", torch.rand(1))