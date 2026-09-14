import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


runs = {
    "lr=0.001": Path("outputs/seed0_lr0001"),
    "lr=0.01": Path("outputs/seed0_lr001"),
    "lr=0.1": Path("outputs/seed0_lr01"),
}


plt.figure()

for label, run_dir in runs.items():

    with open(
        run_dir / "history.json",
        "r",
        encoding="utf-8"
    ) as f:
        history = json.load(f)

    val_loss = history["val_loss"]

    plt.plot(
        range(len(val_loss)),
        val_loss,
        label=label
    )


plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/lr_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("saved: outputs/lr_comparison.png")