import json
from pathlib import Path

import numpy as np

from data import make_dataset


numpy_runs = [
    Path("outputs/numpy_seed0_lr001"),
    Path("outputs/numpy_seed1_lr001"),
    Path("outputs/numpy_seed2_lr001"),
]

torch_runs = [
    Path("outputs/seed0_lr001"),
    Path("outputs/seed1_lr001"),
    Path("outputs/seed2_lr001"),
]


# -------------------------
# 1. baseline
# -------------------------
dataset = make_dataset(
    n_samples=2000,
    data_seed=42
)

_, y_train = dataset["train"]
_, y_test = dataset["test"]

train_mean = np.mean(y_train)

baseline_mse = np.mean(
    (train_mean - y_test) ** 2
)


# -------------------------
# 2. NumPy results
# -------------------------
numpy_mses = []

for run_dir in numpy_runs:
    with open(
        run_dir / "metrics.json",
        "r",
        encoding="utf-8"
    ) as f:
        metrics = json.load(f)

    numpy_mses.append(
        metrics["test_mse"]
    )


# -------------------------
# 3. PyTorch results
# -------------------------
torch_mses = []

for run_dir in torch_runs:
    with open(
        run_dir / "eval_metrics.json",
        "r",
        encoding="utf-8"
    ) as f:
        metrics = json.load(f)

    torch_mses.append(
        metrics["test_mse"]
    )


numpy_mses = np.array(numpy_mses)
torch_mses = np.array(torch_mses)


# -------------------------
# 4. 汇总
# -------------------------
print("Baseline test MSE:")
print(f"  {baseline_mse:.6f}")

print()

print("NumPy:")
print("  per seed =", numpy_mses)
print(f"  mean = {numpy_mses.mean():.6f}")
print(f"  std  = {numpy_mses.std(ddof=1):.6f}")

print()

print("PyTorch:")
print("  per seed =", torch_mses)
print(f"  mean = {torch_mses.mean():.6f}")
print(f"  std  = {torch_mses.std(ddof=1):.6f}")

print()

print("Final comparison:")
print(f"  Baseline : {baseline_mse:.6f}")
print(
    f"  NumPy    : "
    f"{numpy_mses.mean():.6f} ± "
    f"{numpy_mses.std(ddof=1):.6f}"
)
print(
    f"  PyTorch  : "
    f"{torch_mses.mean():.6f} ± "
    f"{torch_mses.std(ddof=1):.6f}"
)