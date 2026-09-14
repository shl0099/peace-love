# Tensor 到参数更新：PyTorch 最小训练流程

## 1. Tensor

PyTorch 中的数据和模型参数都可以表示为 Tensor。

例如：

    x = torch.tensor([1.0, 2.0, 3.0])

如果某个 Tensor 是需要训练的参数，可以设置：

    w = torch.tensor(
        1.0,
        requires_grad=True
    )

`requires_grad=True` 表示 PyTorch 需要记录与该 Tensor 相关的运算，从而在之后计算梯度。


## 2. Forward：从参数得到预测

以线性回归为例：

    y_pred = w * x + b

当前的 w 和 b 决定模型输出。

随后计算 loss：

    loss = torch.mean(
        (y_pred - y_true) ** 2
    )

此时可以理解为：

    parameters
        ↓
    forward
        ↓
    prediction
        ↓
    loss


## 3. Computation Graph

当参与运算的参数具有：

    requires_grad=True

PyTorch 会记录这些运算之间的关系，形成计算图。

例如：

    w, b
      ↓
    y_pred = w*x+b
      ↓
    MSE
      ↓
    loss

计算图使 PyTorch 能够知道：

    loss

是如何由：

    w
    b

计算得到的。


## 4. zero_grad：清除旧梯度

PyTorch 默认会累积梯度。

因此每次进行新的参数更新之前需要：

    optimizer.zero_grad()

它会清除上一轮保存在：

    w.grad
    b.grad

中的梯度。

如果忘记清除，当前梯度会和过去的梯度累加。


## 5. backward：反向传播

执行：

    loss.backward()

PyTorch 会沿计算图从 loss 反向计算：

    ∂loss/∂w
    ∂loss/∂b

并把结果保存到：

    w.grad
    b.grad

因此：

    backward()

负责计算梯度，但它本身并不会修改 w 和 b。


## 6. Gradient Check

更新参数之前可以检查：

    assert w.grad is not None
    assert b.grad is not None

确保梯度确实存在。

还可以检查：

    assert torch.isfinite(w.grad).all()
    assert torch.isfinite(b.grad).all()

确保梯度没有出现：

    NaN
    Inf

在本周 lr=0.2 的失败实验中，这个检查成功捕获了数值失稳。


## 7. optimizer.step：更新参数

执行：

    optimizer.step()

优化器会利用当前梯度修改参数。

对于最基本的 SGD，可以近似理解为：

    w_new = w_old - lr * w.grad
    b_new = b_old - lr * b.grad

其中：

    lr

表示 learning rate。

learning rate 太小：

    参数每次移动距离较小
    收敛可能很慢

learning rate 合适：

    参数稳定向较优位置移动

learning rate 过大：

    参数可能越过最优区域
    产生震荡甚至数值发散


## 8. 一次完整训练步骤

标准 PyTorch 训练循环的核心是：

    y_pred = w * x + b

    loss = torch.mean(
        (y_pred - y) ** 2
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

整体过程：

    Tensor / Parameters
            ↓
         Forward
            ↓
       Prediction
            ↓
          Loss
            ↓
       zero_grad
            ↓
        backward
            ↓
        Gradient
            ↓
     optimizer.step
            ↓
    Updated Parameters
            ↓
       Next Epoch


## 9. NumPy 与 PyTorch 的对应关系

NumPy 中需要手动计算：

    dw = 2 * mean(error * x)
    db = 2 * mean(error)

然后手动更新：

    w = w - lr * dw
    b = b - lr * db

PyTorch 中对应为：

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

因此 PyTorch 并没有改变梯度下降的基本原理。

它主要通过 autograd 自动完成复杂模型中的梯度计算。


## 10. 本周实验中的实际验证

固定真实模型：

    y = 3x + 2 + noise

PyTorch 最终得到：

    w ≈ 3.0037
    b ≈ 2.0169

test MSE：

    ≈ 0.2543

train-mean baseline MSE：

    ≈ 75.8387

checkpoint 重新加载后的固定输入预测差：

    0.0

说明：

1. forward 和 backward 正常工作；
2. optimizer 能正确更新参数；
3. checkpoint 可以恢复模型；
4. 模型确实学到了数据中的线性关系。
