import argparse

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import TwoDimDataset
from model import TwoDimMLP


# =========================
# 1. 命令行参数
# =========================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--train-seed",
    type=int,
    default=0
)

parser.add_argument(
    "--optimizer",
    type=str,
    choices=["sgd", "adam"],
    default="sgd"
)

args = parser.parse_args()

train_seed = args.train_seed
optimizer_name = args.optimizer

print(
    "optimizer =", optimizer_name,
    "train_seed =", train_seed
)


# =========================
# 2. 固定训练随机性
# =========================
torch.manual_seed(train_seed)


# =========================
# 3. 创建固定数据集
# =========================
dataset = TwoDimDataset(
    n_samples=1000,
    seed=42
)


# =========================
# 4. 固定 train / val / test 划分
# =========================
split_g = torch.Generator()
split_g.manual_seed(2026)

train_set, val_set, test_set = random_split(
    dataset,
    [700, 150, 150],
    generator=split_g
)


# =========================
# 5. DataLoader
# =========================
train_g = torch.Generator()
train_g.manual_seed(train_seed)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    generator=train_g
)

val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False
)


# =========================
# 6. 模型和 loss
# =========================
model = TwoDimMLP()

criterion = nn.CrossEntropyLoss()

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print("num_params =", num_params)

# =========================
# 7. optimizer
# =========================
if optimizer_name == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1
    )

else:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )


# =========================
# 8. checkpoint
# =========================
checkpoint_path = (
    f"best_model_{optimizer_name}_seed{train_seed}.pt"
)

best_val_loss = float("inf")

start_time = time.perf_counter()
# =========================
# 9. 训练
# =========================
for epoch in range(10):

    # -------------------------
    # Train
    # -------------------------
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_samples = 0

    for x_batch, y_batch in train_loader:

        # forward
        logits = model(x_batch)

        loss = criterion(
            logits,
            y_batch
        )

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy
        pred = logits.argmax(dim=1)

        train_loss += (
            loss.item() * len(y_batch)
        )

        train_correct += (
            (pred == y_batch)
            .sum()
            .item()
        )

        train_samples += len(y_batch)

    avg_train_loss = (
        train_loss / train_samples
    )

    train_accuracy = (
        train_correct / train_samples
    )


    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():

        for x_batch, y_batch in val_loader:

            logits = model(x_batch)

            loss = criterion(
                logits,
                y_batch
            )

            pred = logits.argmax(dim=1)

            val_loss += (
                loss.item() * len(y_batch)
            )

            val_correct += (
                (pred == y_batch)
                .sum()
                .item()
            )

            val_samples += len(y_batch)

    avg_val_loss = (
        val_loss / val_samples
    )

    val_accuracy = (
        val_correct / val_samples
    )


    # -------------------------
    # 保存验证集最优模型
    # -------------------------
    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            "  -> saved best model:",
            checkpoint_path
        )


    # -------------------------
    # 输出当前 epoch
    # -------------------------
    print(
        "epoch =", epoch,
        "train_loss =", round(avg_train_loss, 4),
        "train_acc =", round(train_accuracy, 4),
        "val_loss =", round(avg_val_loss, 4),
        "val_acc =", round(val_accuracy, 4)
    )

elapsed_time = time.perf_counter() - start_time

print("training_time_seconds =", elapsed_time)
