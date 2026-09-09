"""
Week 1: NumPy Linear Regression

功能：
1. 生成一元线性回归数据 y = 3x + 2 + noise
2. 按 70% / 15% / 15% 划分训练集、验证集、测试集
3. 使用 NumPy 和梯度下降从零训练线性回归
4. 比较多个 learning rate
5. 使用验证集选择最佳 learning rate
6. 在测试集上进行最终评估
7. 与“预测训练集均值”的 baseline 对比
8. 保存 loss 曲线和实验指标

运行示例：
python first_week_script.py

python first_week_script.py \
    --n_samples 800 \
    --noise 0.3 \
    --epochs 2000 \
    --seed 123 \
    --learning_rates 0.001 0.01 0.1
"""

from pathlib import Path
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 全局常量
# ============================================================

W_TRUE = 3.0
B_TRUE = 2.0


# ============================================================
# 1. 命令行参数
# ============================================================

def parse_args():
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="NumPy linear regression experiment."
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of generated samples."
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.5,
        help="Standard deviation of Gaussian noise."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1],
        help="Learning rates to compare."
    )

    return parser.parse_args()


# ============================================================
# 2. 数据生成与划分
# ============================================================

def generate_data(n_samples, noise_std, seed):
    """
    生成线性回归数据并划分数据集。

    数据模型：
        y = 3x + 2 + noise

    Args:
        n_samples (int):
            样本总数。

        noise_std (float):
            高斯噪声标准差。

        seed (int):
            随机种子。

    Returns:
        x_train, y_train,
        x_val, y_val,
        x_test, y_test
    """

    np.random.seed(seed)

    # ---------- 生成数据 ----------
    x = np.random.uniform(
        low=-5.0,
        high=5.0,
        size=n_samples
    )

    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=n_samples
    )

    y = W_TRUE * x + B_TRUE + noise

    # ---------- 随机打乱 ----------
    indices = np.random.permutation(n_samples)

    x = x[indices]
    y = y[indices]

    # ---------- 计算划分位置 ----------
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    # ---------- 划分数据 ----------
    x_train = x[:train_end]
    y_train = y[:train_end]

    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]

    x_test = x[val_end:]
    y_test = y[val_end:]

    # ---------- 基本检查 ----------
    assert x_train.shape == y_train.shape
    assert x_val.shape == y_val.shape
    assert x_test.shape == y_test.shape

    assert len(x_train) > 0
    assert len(x_val) > 0
    assert len(x_test) > 0

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    )


# ============================================================
# 3. 模型训练
# ============================================================

def train_linear_regression(
    x_train,
    y_train,
    learning_rate,
    epochs
):
    """
    使用梯度下降训练一元线性回归模型。

    模型：
        y_pred = w * x + b

    Args:
        x_train (np.ndarray):
            训练输入。

        y_train (np.ndarray):
            训练标签。

        learning_rate (float):
            学习率。

        epochs (int):
            训练轮数。

    Returns:
        w (float):
            学习得到的权重。

        b (float):
            学习得到的偏置。

        losses (list):
            每个 epoch 的训练 MSE。
    """

    # ---------- 参数初始化 ----------
    w = 0.0
    b = 0.0

    losses = []

    # ---------- 梯度下降 ----------
    for _ in range(epochs):

        # 1. 前向预测
        y_pred = w * x_train + b

        # 2. 计算误差
        error = y_pred - y_train

        # 3. MSE Loss
        loss = np.mean(error ** 2)

        # 4. 计算梯度
        dw = 2.0 * np.mean(error * x_train)
        db = 2.0 * np.mean(error)

        # 5. 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db

        # 6. 保存 loss
        losses.append(float(loss))

    # ---------- 训练检查 ----------
    assert len(losses) == epochs

    assert np.isfinite(w)
    assert np.isfinite(b)

    assert np.all(np.isfinite(losses))

    assert losses[-1] < losses[0], (
        f"Loss did not decrease for lr={learning_rate}"
    )

    return float(w), float(b), losses


# ============================================================
# 4. 模型评估
# ============================================================

def evaluate(x, y, w, b):
    """
    计算模型在给定数据上的 MSE。

    Args:
        x (np.ndarray):
            输入数据。

        y (np.ndarray):
            真实标签。

        w (float):
            模型权重。

        b (float):
            模型偏置。

    Returns:
        mse (float):
            均方误差。
    """

    y_pred = w * x + b

    mse = np.mean(
        (y_pred - y) ** 2
    )

    return float(mse)


# ============================================================
# 5. 选择最佳模型
# ============================================================

def select_best_model(results):
    """
    根据验证集 MSE 选择最佳 learning rate。

    Args:
        results (dict):
            不同 learning rate 的实验结果。

    Returns:
        best_lr
        best_w
        best_b
    """

    best_lr = min(
        results,
        key=lambda lr: results[lr]["val_mse"]
    )

    best_w = results[best_lr]["w"]
    best_b = results[best_lr]["b"]

    return best_lr, best_w, best_b


# ============================================================
# 6. Loss 曲线
# ============================================================

def plot_losses(all_losses, output_dir):
    """
    绘制不同 learning rate 的训练 loss 曲线。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))

    for lr, losses in all_losses.items():
        plt.plot(
            losses,
            label=f"lr={lr}"
        )
    '''
    for key, value in dictionary.items():  这个python模式很常用
    results = {
        "loss": 0.25,
        "accuracy": 0.95,
        "runtime": 2.3
    }

    for name, value in results.items():
        print(name, value)

    name="loss"      value=0.25
    name="accuracy"  value=0.95
    name="runtime"   value=2.3

    '''

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = (
        output_dir /
        "loss_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    assert output_path.exists()


# ============================================================
# 7. 保存实验指标
# ============================================================

def save_metrics(
    results,
    best_lr,
    best_w,
    best_b,
    test_mse,
    baseline_mse,
    w_relative_error,
    b_relative_error,
    runtime,
    output_dir
):
    """
    将实验结果保存到 metrics.txt。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path = (
        output_dir /
        "metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "========== Experiment Results ==========\n\n"
        )

        # ---------- 每个 lr ----------
        for lr, result in results.items():

            f.write(
                f"learning_rate = {lr}\n"
            )

            f.write(
                f"w = {result['w']:.6f}\n"
            )

            f.write(
                f"b = {result['b']:.6f}\n"
            )

            f.write(
                f"val_mse = {result['val_mse']:.6f}\n\n"
            )

        # ---------- 最佳模型 ----------
        f.write(
            "========== Best Model ==========\n"
        )

        f.write(
            f"best_lr = {best_lr}\n"
        )

        f.write(
            f"best_w = {best_w:.6f}\n"
        )

        f.write(
            f"best_b = {best_b:.6f}\n"
        )

        f.write(
            f"test_mse = {test_mse:.6f}\n"
        )

        f.write(
            f"baseline_mse = {baseline_mse:.6f}\n"
        )

        f.write(
            f"w_relative_error = "
            f"{w_relative_error * 100:.2f}%\n"
        )

        f.write(
            f"b_relative_error = "
            f"{b_relative_error * 100:.2f}%\n"
        )

        f.write(
            f"runtime = {runtime:.6f} seconds\n"
        )

    assert metrics_path.exists()


# ============================================================
# 8. 主实验流程
# ============================================================

def main():

    # ---------- 读取参数 ----------
    args = parse_args()

    # ---------- 参数合法性检查 ----------
    assert args.n_samples >= 10
    assert args.epochs > 0
    assert args.noise >= 0

    assert all(
        lr > 0
        for lr in args.learning_rates
    )

    # ---------- 输出目录 ----------
    output_dir = Path("results")

    # ---------- 生成数据 ----------
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    ) = generate_data(
        n_samples=args.n_samples,
        noise_std=args.noise,
        seed=args.seed
    )

    print(
        f"train shape: {x_train.shape}"
    )

    print(
        f"val shape:   {x_val.shape}"
    )

    print(
        f"test shape:  {x_test.shape}"
    )

    # ---------- 保存实验结果 ----------
    all_losses = {}
    results = {}

    # ---------- 开始计时 ----------
    start_time = time.time()

    # ========================================================
    # 比较不同 learning rate
    # ========================================================

    for lr in args.learning_rates:

        w, b, losses = train_linear_regression(
            x_train=x_train,
            y_train=y_train,
            learning_rate=lr,
            epochs=args.epochs
        )

        val_mse = evaluate(
            x=x_val,
            y=y_val,
            w=w,
            b=b
        )

        all_losses[lr] = losses

        results[lr] = {
            "w": w,
            "b": b,
            "val_mse": val_mse
        }

        print(
            f"\nlr = {lr}"
        )

        print(
            f"w = {w:.6f}"
        )

        print(
            f"b = {b:.6f}"
        )

        print(
            f"val_mse = {val_mse:.6f}"
        )

    # ---------- 结束计时 ----------
    runtime = (
        time.time() -
        start_time
    )

    # ========================================================
    # 根据验证集选择最佳模型
    # ========================================================

    (
        best_lr,
        best_w,
        best_b
    ) = select_best_model(
        results
    )

    # ========================================================
    # 测试集最终评估
    # ========================================================

    test_mse = evaluate(
        x=x_test,
        y=y_test,
        w=best_w,
        b=best_b
    )

    # ========================================================
    # Mean Baseline
    # ========================================================

    mean_prediction = np.mean(
        y_train
    )

    baseline_mse = np.mean(
        (mean_prediction - y_test) ** 2
    )

    baseline_mse = float(
        baseline_mse
    )

    # 模型应该明显优于简单 baseline
    assert test_mse < baseline_mse

    # ========================================================
    # 参数相对误差
    # ========================================================

    w_relative_error = (
        abs(best_w - W_TRUE)
        / abs(W_TRUE)
    )

    b_relative_error = (
        abs(best_b - B_TRUE)
        / abs(B_TRUE)
    )

    # ========================================================
    # 输出结果
    # ========================================================

    print(
        "\n========== Best Model =========="
    )

    print(
        f"best lr = {best_lr}"
    )

    print(
        f"best w = {best_w:.6f}"
    )

    print(
        f"best b = {best_b:.6f}"
    )

    print(
        f"test mse = {test_mse:.6f}"
    )

    print(
        f"baseline mse = {baseline_mse:.6f}"
    )

    print(
        f"w relative error = "
        f"{w_relative_error * 100:.2f}%"
    )

    print(
        f"b relative error = "
        f"{b_relative_error * 100:.2f}%"
    )

    print(
        f"runtime = {runtime:.6f} seconds"
    )

    # ========================================================
    # 保存实验结果
    # ========================================================

    plot_losses(
        all_losses=all_losses,
        output_dir=output_dir
    )

    save_metrics(
        results=results,
        best_lr=best_lr,
        best_w=best_w,
        best_b=best_b,
        test_mse=test_mse,
        baseline_mse=baseline_mse,
        w_relative_error=w_relative_error,
        b_relative_error=b_relative_error,
        runtime=runtime,
        output_dir=output_dir
    )

    print(
        "\nResults saved to:",
        output_dir.resolve()
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()