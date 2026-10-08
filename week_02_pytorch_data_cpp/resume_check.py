import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import TwoDimDataset
from model import TwoDimMLP


DATA_SEED = 42
SPLIT_SEED = 2026
TRAIN_SEED = 0

BATCH_SIZE = 32
LR = 0.01

CHECKPOINT_PATH = "resume_checkpoint.pt"


def make_dataset():
    dataset = TwoDimDataset(
        n_samples=1000,
        seed=DATA_SEED
    )

    split_g = torch.Generator()
    split_g.manual_seed(SPLIT_SEED)

    train_set, val_set, test_set = random_split(
        dataset,
        [700, 150, 150],
        generator=split_g
    )

    return train_set, val_set, test_set


def make_train_loader(train_set, generator):
    return DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )


def train_one_epoch(
    model,
    optimizer,
    criterion,
    train_loader
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in train_loader:

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * len(y_batch)
        )

        total_samples += len(y_batch)

    return total_loss / total_samples


def predict_fixed(model):
    model.eval()

    fixed_x = torch.tensor([
        [-1.5, 0.0],
        [-0.8, 1.0],
        [0.8, -1.0],
        [1.5, 0.0],
    ])

    with torch.no_grad():
        logits = model(fixed_x)

    return logits


# =========================================================
# A. uninterrupted：连续训练 10 epoch
# =========================================================

train_set, _, _ = make_dataset()

torch.manual_seed(TRAIN_SEED)

model_a = TwoDimMLP()

optimizer_a = torch.optim.Adam(
    model_a.parameters(),
    lr=LR
)

criterion = nn.CrossEntropyLoss()

train_g_a = torch.Generator()
train_g_a.manual_seed(TRAIN_SEED)

train_loader_a = make_train_loader(
    train_set,
    train_g_a
)

print("=== uninterrupted ===")

for epoch in range(10):

    loss = train_one_epoch(
        model_a,
        optimizer_a,
        criterion,
        train_loader_a
    )

    print(
        "epoch =",
        epoch,
        "loss =",
        loss
    )

pred_a = predict_fixed(model_a)


# =========================================================
# B. interrupted：训练 5 epoch 后保存
# =========================================================

train_set, _, _ = make_dataset()

torch.manual_seed(TRAIN_SEED)

model_b = TwoDimMLP()

optimizer_b = torch.optim.Adam(
    model_b.parameters(),
    lr=LR
)

train_g_b = torch.Generator()
train_g_b.manual_seed(TRAIN_SEED)

train_loader_b = make_train_loader(
    train_set,
    train_g_b
)

print("\n=== before interruption ===")

for epoch in range(5):

    loss = train_one_epoch(
        model_b,
        optimizer_b,
        criterion,
        train_loader_b
    )

    print(
        "epoch =",
        epoch,
        "loss =",
        loss
    )


# =========================================================
# C. 保存完整训练现场
# =========================================================

checkpoint = {
    "epoch": 4,

    "model_state":
        model_b.state_dict(),

    "optimizer_state":
        optimizer_b.state_dict(),

    "torch_rng_state":
        torch.get_rng_state(),

    "train_loader_rng_state":
        train_g_b.get_state(),
}

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

print(
    "\nsaved checkpoint:",
    CHECKPOINT_PATH
)


# =========================================================
# D. 模拟程序完全重启
# =========================================================

model_c = TwoDimMLP()

optimizer_c = torch.optim.Adam(
    model_c.parameters(),
    lr=LR
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model_c.load_state_dict(
    checkpoint["model_state"]
)

optimizer_c.load_state_dict(
    checkpoint["optimizer_state"]
)

torch.set_rng_state(
    checkpoint["torch_rng_state"]
)

train_g_c = torch.Generator()

train_g_c.set_state(
    checkpoint["train_loader_rng_state"]
)


train_set, _, _ = make_dataset()

train_loader_c = make_train_loader(
    train_set,
    train_g_c
)

start_epoch = (
    checkpoint["epoch"] + 1
)


# =========================================================
# E. resume：继续 epoch 5 ~ 9
# =========================================================

print("\n=== resumed ===")

for epoch in range(
    start_epoch,
    10
):

    loss = train_one_epoch(
        model_c,
        optimizer_c,
        criterion,
        train_loader_c
    )

    print(
        "epoch =",
        epoch,
        "loss =",
        loss
    )

pred_c = predict_fixed(model_c)


# =========================================================
# F. uninterrupted vs resumed
# =========================================================

max_diff = (
    pred_a - pred_c
).abs().max().item()

print("\n=== comparison ===")

print("uninterrupted logits =")
print(pred_a)

print("\nresumed logits =")
print(pred_c)

print(
    "\nmax prediction diff =",
    max_diff
)

print(
    "pass <= 1e-5 =",
    max_diff <= 1e-5
)
