import torch
import torch.nn as nn


class TwoDimMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)  #得到[batch, 8]
        x = self.relu(x)  #shape不变 但是里面所有的负数全变成0
        logits = self.fc2(x)
        return logits
        return x
