import argparse

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
    required=True
)

parser.add_argument(
    "--optimizer",
    type=str,
    choices=["sgd", "adam"],
    required=True
)

args = parser.parse_args()

train_seed = args.train_seed
optimizer_name = args.optimizer


# =========================
# 2. 重建相同数据集
# =========================
dataset = TwoDimDataset(
    n_samples=1000,
    seed=42
)


# =========================
# 3. 重建相同数据划分
# =========================
split_g = torch.Generator()
split_g.manual_seed(2026)

train_set, val_set, test_set = random_split(
    dataset,
    [700, 150, 150],
    generator=split_g
)


# =========================
# 4. Test DataLoader
# =========================
test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)


# =========================
# 5. 创建模型
# =========================
model = TwoDimMLP()


# =========================
# 6. 加载 best checkpoint
# =========================
checkpoint_path = (
    f"best_model_{optimizer_name}_seed{train_seed}.pt"
)

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()


# =========================
# 7. Test
# =========================
criterion = nn.CrossEntropyLoss()

total_loss = 0.0
total_correct = 0
total_samples = 0

with torch.no_grad():

    for x_batch, y_batch in test_loader:

        logits = model(x_batch)

        loss = criterion(
            logits,
            y_batch
        )

        pred = logits.argmax(dim=1)

        total_loss += (
            loss.item() * len(y_batch)
        )

        total_correct += (
            (pred == y_batch)
            .sum()
            .item()
        )

        total_samples += len(y_batch)


test_loss = total_loss / total_samples
test_accuracy = total_correct / total_samples


# =========================
# 8. 输出
# =========================
print("optimizer =", optimizer_name)
print("train_seed =", train_seed)
print("checkpoint =", checkpoint_path)
print("test_loss =", test_loss)
print("test_accuracy =", test_accuracy)
print("test_samples =", total_samples)
