import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from data import make_dataset
from utils import set_seed, get_device, make_output_dir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/run_seed0"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------
    # 1. 基础设置
    # -------------------------
    set_seed(args.seed)

    device = get_device(args.device)
    output_dir = make_output_dir(args.output_dir)

    print("seed =", args.seed)
    print("lr =", args.lr)
    print("device =", device)
    print("output_dir =", output_dir)

    # -------------------------
    # 2. 数据
    # -------------------------
    data_seed = 42

    dataset = make_dataset(
        n_samples=2000,
        data_seed=data_seed
    )

    x_train, y_train = dataset["train"]
    x_val, y_val = dataset["val"]

    x_train = torch.tensor(
        x_train,
        dtype=torch.float32,
        device=device
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=device
    )

    x_val = torch.tensor(
        x_val,
        dtype=torch.float32,
        device=device
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.float32,
        device=device
    )

    # -------------------------
    # 3. 参数初始化
    # -------------------------
    w = torch.randn(
        (),
        device=device,
        requires_grad=True
    )

    b = torch.randn(
        (),
        device=device,
        requires_grad=True
    )

    print("initial w =", w.item())
    print("initial b =", b.item())

    optimizer = torch.optim.SGD(
        [w, b],
        lr=args.lr
    )

    train_losses = []
    val_losses = []

    start_time = time.perf_counter()

    # -------------------------
    # 4. 训练
    # -------------------------
    for epoch in range(args.epochs):

        # forward
        y_pred = w * x_train + b

        loss = torch.mean(
            (y_pred - y_train) ** 2
        )

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # gradient checks
        assert w.grad is not None
        assert b.grad is not None

        assert torch.isfinite(w.grad).all()
        assert torch.isfinite(b.grad).all()

        # update parameters
        optimizer.step()

        # -------------------------
        # 用更新后的同一组参数
        # 同时计算 train / val loss
        # -------------------------
        with torch.no_grad():

            train_pred = w * x_train + b

            train_loss = torch.mean(
                (train_pred - y_train) ** 2
            )

            val_pred = w * x_val + b

            val_loss = torch.mean(
                (val_pred - y_val) ** 2
            )

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch={epoch:3d} "
                f"train_loss={train_loss.item():.6f} "
                f"val_loss={val_loss.item():.6f} "
                f"w={w.item():.4f} "
                f"b={b.item():.4f}"
            )

    runtime = time.perf_counter() - start_time

    # -------------------------
    # 5. 保存 checkpoint
    # -------------------------
    checkpoint_path = output_dir / "model.pt"
    # 固定输入，用于以后验证 save / reload 一致性
    fixed_x = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        fixed_pred_before_save = (
                w * fixed_x + b
        ).detach().cpu()

    torch.save(
        {
            "w": w.detach().cpu(),
            "b": b.detach().cpu(),
            "fixed_x": fixed_x.detach().cpu(),
            "fixed_pred_before_save": fixed_pred_before_save,
        },
        checkpoint_path
    )



    # -------------------------
    # 6. 保存 config
    # -------------------------
    config = {
        "seed": args.seed,
        "data_seed": data_seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "device": str(device),
        "n_samples": 2000,
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
    # 7. 保存 metrics
    # -------------------------
    metrics = {
        "final_train_mse": train_losses[-1],
        "final_val_mse": val_losses[-1],
        "final_w": w.item(),
        "final_b": b.item(),
        "runtime_seconds": runtime,
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

    # -------------------------
    # 8. 保存 loss 历史
    # -------------------------
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }

    with open(
        output_dir / "history.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            history,
            f
        )

    # -------------------------
    # 9. 保存 loss curve
    # -------------------------
    plt.figure()

    plt.plot(
        train_losses,
        label="train"
    )

    plt.plot(
        val_losses,
        label="validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\ntraining finished")
    print("final w =", w.item())
    print("final b =", b.item())
    print("runtime =", runtime)
    print("saved to =", output_dir)


if __name__ == "__main__":
    main()