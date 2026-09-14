import json
from pathlib import Path

import numpy as np


run_dirs = [
    Path("outputs/seed0_lr001"),
    Path("outputs/seed1_lr001"),
    Path("outputs/seed2_lr001"),
]

test_mses = []

for run_dir in run_dirs:
    metrics_path = run_dir / "eval_metrics.json"

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as f:
        metrics = json.load(f)

    test_mse = metrics["test_mse"]
    test_mses.append(test_mse)

    print(
        run_dir.name,
        "test_mse =",
        test_mse
    )


test_mses = np.array(test_mses)

mean_mse = np.mean(test_mses)

std_mse = np.std(
    test_mses,
    ddof=1
)

print()
print("mean test MSE =", mean_mse)
print("std test MSE  =", std_mse)
print(
    f"mean ± std = "
    f"{mean_mse:.6f} ± {std_mse:.6f}"
)