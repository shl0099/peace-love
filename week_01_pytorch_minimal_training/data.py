import numpy as np


def make_dataset(
    n_samples=2000,
    data_seed=42,
    true_w=3.0,
    true_b=2.0,
    noise_std=0.5,
):
    rng = np.random.default_rng(data_seed)

    # 生成输入
    x = rng.uniform(-5.0, 5.0, size=n_samples)

    # 生成噪声
    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=n_samples
    )

    # y = 3x + 2 + noise
    y = true_w * x + true_b + noise

    # 按样本 ID 划分，而不是重新随机抽
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)

    train_end = n_train
    val_end = n_train + n_val

    x_train = x[:train_end]
    y_train = y[:train_end]

    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]

    x_test = x[val_end:]
    y_test = y[val_end:]

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


if __name__ == "__main__":
    dataset = make_dataset()

    for split_name, (x, y) in dataset.items():
        print(
            split_name,
            "x shape =", x.shape,
            "y shape =", y.shape
        )