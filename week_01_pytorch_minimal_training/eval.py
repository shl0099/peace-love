import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import make_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)

    # -------------------------
    # 1. 加载训练配置
    # -------------------------
    with open(
        run_dir / "config.json",
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    print("run_dir =", run_dir)
    print("training seed =", config["seed"])
    print("data seed =", config["data_seed"])

    # -------------------------
    # 2. 重新生成同一份数据
    # -------------------------
    dataset = make_dataset(
        n_samples=config["n_samples"],
        data_seed=config["data_seed"]
    )

    x_train, y_train = dataset["train"]
    x_test, y_test = dataset["test"]

    # eval 固定在 CPU
    x_test = torch.tensor(
        x_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )

    # -------------------------
    # 3. 加载 checkpoint
    # -------------------------
    checkpoint = torch.load(
        run_dir / "model.pt",
        map_location="cpu"
    )

    w = checkpoint["w"]
    b = checkpoint["b"]

    print("loaded w =", w.item())
    print("loaded b =", b.item())

    # -------------------------
    # 4. 测试 save / reload 一致性
    # -------------------------
    fixed_x = checkpoint["fixed_x"]
    pred_before_save = checkpoint[
        "fixed_pred_before_save"
    ]

    with torch.no_grad():
        pred_after_reload = (
            w * fixed_x + b
        )

    max_abs_diff = torch.max(
        torch.abs(
            pred_before_save - pred_after_reload
        )
    ).item()

    print(
        "reload max abs diff =",
        max_abs_diff
    )

    assert max_abs_diff <= 1e-6

    # -------------------------
    # 5. 模型 test MSE
    # -------------------------
    with torch.no_grad():

        test_pred = (
            w * x_test + b
        )

        test_mse = torch.mean(
            (test_pred - y_test) ** 2
        ).item()

    # -------------------------
    # 6. train-mean baseline
    # -------------------------
    train_mean = np.mean(y_train)

    baseline_pred = torch.full_like(
        y_test,
        fill_value=float(train_mean)
    )

    baseline_mse = torch.mean(
        (baseline_pred - y_test) ** 2
    ).item()

    print("test MSE =", test_mse)
    print(
        "train-mean baseline MSE =",
        baseline_mse
    )

    # 模型必须优于 baseline
    assert test_mse < baseline_mse

    # -------------------------
    # 7. 保存独立评估结果
    # -------------------------
    eval_metrics = {
        "test_mse": test_mse,
        "baseline_test_mse": baseline_mse,
        "reload_max_abs_diff": max_abs_diff,
        "model_beats_baseline": (
            test_mse < baseline_mse
        ),
    }

    with open(
        run_dir / "eval_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            eval_metrics,
            f,
            indent=4
        )

    print("\nevaluation passed")
    print(
        "saved:",
        run_dir / "eval_metrics.json"
    )


if __name__ == "__main__":
    main()