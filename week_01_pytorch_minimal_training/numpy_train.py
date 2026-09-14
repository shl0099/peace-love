import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import make_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/numpy_seed0_lr001"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------
    # 1. 创建输出目录
    # -------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------
    # 2. 固定训练随机性
    # -------------------------
    rng = np.random.default_rng(args.seed)

    # -------------------------
    # 3. 获取与 PyTorch 完全相同的数据
    # -------------------------
    data_seed = 42

    dataset = make_dataset(
        n_samples=2000,
        data_seed=data_seed
    )

    x_train, y_train = dataset["train"]
    x_val, y_val = dataset["val"]
    x_test, y_test = dataset["test"]

    # -------------------------
    # 4. 随机初始化参数
    # -------------------------
    w = rng.normal()
    b = rng.normal()

    print("seed =", args.seed)
    print("lr =", args.lr)
    print("initial w =", w)
    print("initial b =", b)

    train_losses = []
    val_losses = []

    start_time = time.perf_counter()

    # -------------------------
    # 5. NumPy 手写梯度下降
    # -------------------------
    for epoch in range(args.epochs):

        # forward
        y_pred = w * x_train + b

        error = y_pred - y_train

        loss = np.mean(
            error ** 2
        )

        # 手动计算梯度
        dw = 2.0 * np.mean(
            error * x_train
        )

        db = 2.0 * np.mean(
            error
        )

        # 参数更新
        w = w - args.lr * dw
        b = b - args.lr * db

        # 用更新后的参数重新计算 train / val
        train_pred = w * x_train + b
        val_pred = w * x_val + b

        train_loss = np.mean(
            (train_pred - y_train) ** 2
        )

        val_loss = np.mean(
            (val_pred - y_val) ** 2
        )

        train_losses.append(
            float(train_loss)
        )

        val_losses.append(
            float(val_loss)
        )

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch={epoch:3d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"w={w:.4f} "
                f"b={b:.4f}"
            )

    runtime = time.perf_counter() - start_time

    # -------------------------
    # 6. test MSE
    # -------------------------
    test_pred = w * x_test + b

    test_mse = np.mean(
        (test_pred - y_test) ** 2
    )

    # -------------------------
    # 7. train-mean baseline
    # -------------------------
    train_mean = np.mean(y_train)

    baseline_pred = np.full_like(
        y_test,
        train_mean
    )

    baseline_mse = np.mean(
        (baseline_pred - y_test) ** 2
    )

    # -------------------------
    # 8. 保存 config
    # -------------------------
    config = {
        "seed": args.seed,
        "data_seed": data_seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_samples": 2000,
        "framework": "numpy"
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=4
        )

    # -------------------------
    # 9. 保存 metrics
    # -------------------------
    metrics = {
        "final_train_mse": float(train_losses[-1]),
        "final_val_mse": float(val_losses[-1]),
        "test_mse": float(test_mse),
        "baseline_test_mse": float(baseline_mse),
        "final_w": float(w),
        "final_b": float(b),
        "runtime_seconds": runtime,
        "model_beats_baseline": bool(
            test_mse < baseline_mse
        )
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\ntraining finished")
    print("final w =", w)
    print("final b =", b)
    print("test MSE =", test_mse)
    print(
        "train-mean baseline MSE =",
        baseline_mse
    )
    print("runtime =", runtime)
    print("saved to =", output_dir)


if __name__ == "__main__":
    main()