import torch
from torch.utils.data import Dataset


class TwoDimDataset(Dataset):
    def __init__(self, n_samples=1000, seed=42):
        g = torch.Generator()
        g.manual_seed(seed)

        n0 = n_samples // 2
        n1 = n_samples - n0

        # class 0
        # x1 ∈ [-2.0, -0.5]
        # x2 ∈ [-1.5,  1.5]
        x0 = torch.rand(n0, 2, generator=g)

        x0[:, 0] = -2.0 + 1.5 * x0[:, 0]
        x0[:, 1] = -1.5 + 3.0 * x0[:, 1]

        # class 1
        # x1 ∈ [0.5, 2.0]
        # x2 ∈ [-1.5, 1.5]
        x1 = torch.rand(n1, 2, generator=g)

        x1[:, 0] = 0.5 + 1.5 * x1[:, 0]
        x1[:, 1] = -1.5 + 3.0 * x1[:, 1]

        y0 = torch.zeros(
            n0,
            dtype=torch.long
        )

        y1 = torch.ones(
            n1,
            dtype=torch.long
        )

        self.x = torch.cat(
            [x0, x1],
            dim=0
        )

        self.y = torch.cat(
            [y0, y1],
            dim=0
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


'''
torch.manual_seed(...)  固定 PyTorch 的随机数状态
torch.randn(...)  torch.randn(4, 2)  4个样本 每个样本两个特征
torch.zeros(4, dtype=torch.long) 值为0 类型为long的张量
torch.ones(4, dtype=torch.long)     值为1 类型为long的张量

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)
batch_size 指定一次训练多少个 x_batch.shape = [batch_size, 2]  四个样本 每个样本两个输入特征
len(loader) 计算的是完成训练需要装填多少次

python - <<'PY'
from torch.utils.data import DataLoader
from data import TwoDimDataset

dataset = TwoDimDataset(n_samples=10, seed=42)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

print("dataset length =", len(dataset))
print("loader length =", len(loader))

for batch_idx, (x_batch, y_batch) in enumerate(loader):
    print()
    print("batch", batch_idx)
    print("x shape =", x_batch.shape)
    print("y shape =", y_batch.shape)
    print("y =", y_batch)

Pytorch 最基础全连接层 nn.Linear(in_features, out_features)
nn.Linear(2, 8)
每个样本：
2 个特征
↓
线性变换
↓
8 个特征

若最后一层nn.Linear(8, 2)
对于一个样本输出[2.3, -0.7] 意思不是[类别0, 类别1] 而是两个类别最原始的分数logits
第 0 类分数 = 2.3
第 1 类分数 = -0.7
推理时pred = logits.argmax(dim=1) 会得到pred = 0  分数谁高推理时选谁


CrossEntropyLoss 输入是 [batch, classes]
logits = torch.tensor([
    [2.0, 0.5],
    [0.2, 1.8],
    [1.5, 0.3]
], requires_grad=True)

y = torch.tensor([0, 1, 0], dtype=torch.long)

logits.shape = [3, 2]
y.shape      = [3]
CrossEntropyLoss 会逐行看 logits，然后用 y 告诉它：
“这一行正确的类别编号是哪一个？”
样本0 logits = [2.0, 0.5]，真实类别 y=0
样本1 logits = [0.2, 1.8]，真实类别 y=1
样本2 logits = [1.5, 0.3]，真实类别 y=0

若logits = [0.2, 1.8]   → 判断正确，而且领先明显 → loss 较小
logits = [1.79, 1.8]  → 也判断正确，但非常犹豫   → loss 会更大

nn.ReLU()
激活函数
输入 < 0 → 变成 0
输入 > 0 → 保持
'''
