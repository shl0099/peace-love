# W01 - PyTorch Minimal Training Project

## 1. 项目目标

本项目完成 W01 的 PyTorch 最小训练工程与 Linux/Git 实践。

使用固定的 2000 个带噪声线性回归样本：

y = 3x + 2 + noise

主要完成：

- NumPy 手写线性回归
- PyTorch autograd 线性回归
- 70% / 15% / 15% 的 train / validation / test 划分
- train-mean baseline 对比
- 3 个独立 training seeds
- 3 个 learning rates 对比
- checkpoint 保存与重新加载
- 独立 eval
- 梯度非空与 finite 检查
- 学习率过大导致训练失败的 E3 实验
- Linux 虚拟环境、日志重定向与进程操作

## 2. 项目结构

    week01/
    ├── data.py
    ├── utils.py
    ├── train.py
    ├── eval.py
    ├── numpy_train.py
    ├── summarize.py
    ├── compare_lr.py
    ├── compare_models.py
    ├── requirements.txt
    ├── environment_freeze.txt
    ├── README.md
    └── outputs/

## 3. 实验环境

当前实验环境：

- WSL2 Ubuntu
- Python 3.14.4
- NumPy 2.5.3
- Matplotlib 3.11.2
- PyTorch 2.14.0+cpu
- Device: CPU

创建虚拟环境：

    python3 -m venv .venv
    source .venv/bin/activate

安装 NumPy 和 Matplotlib：

    python -m pip install numpy==2.5.3 matplotlib==3.11.2

安装 CPU 版 PyTorch：

    python -m pip install torch==2.14.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu

检查环境：

    python -c "import numpy, matplotlib, torch; print('numpy:', numpy.__version__); print('matplotlib:', matplotlib.__version__); print('torch:', torch.__version__)"

## 4. 数据集

固定生成 2000 个样本：

    y = 3x + 2 + noise

固定：

    data_seed = 42

按照样本 ID 划分：

    train = 1400 (70%)
    val   = 300  (15%)
    test  = 300  (15%)

训练过程中不使用 test 集进行参数学习或模型选择。

## 5. PyTorch 训练

示例：

    python train.py \
      --seed 0 \
      --lr 0.01 \
      --device cpu \
      --output-dir outputs/seed0_lr001

独立评估：

    python eval.py \
      --run-dir outputs/seed0_lr001

每次成功实验会产生：

    config.json
    metrics.json
    history.json
    loss_curve.png
    model.pt

独立评估后还会产生：

    eval_metrics.json

## 6. E1：Baseline、NumPy 与 PyTorch 对比

固定条件：

    n_samples = 2000
    data_seed = 42
    lr = 0.01
    epochs = 500
    train_seed = 0, 1, 2

三个 train seed 表示三次独立训练，而不是对同一个模型重复评估。

最终 test MSE：

    Baseline : 75.838660
    NumPy    : 0.254337 ± 0.000003
    PyTorch  : 0.254336 ± 0.000001

其中 baseline 使用训练集标签均值作为固定预测值：

    baseline_prediction = mean(y_train)

不能使用 mean(y_test)，因为这会提前使用测试集信息，造成数据泄漏。

实验结果表明：

1. NumPy 和 PyTorch 都显著优于 train-mean baseline。
2. NumPy 手写梯度与 PyTorch autograd 得到了几乎一致的最终结果。
3. 三个独立训练 seed 的标准差非常小，说明该实验结论对随机初始化较稳定。

运行三 seed 汇总：

    python summarize.py

运行 NumPy / PyTorch / baseline 总对比：

    python compare_models.py


## 7. E2：Learning Rate 对比

固定：

    seed = 0
    data_seed = 42
    epochs = 500
    device = cpu

只改变 learning rate：

    lr = 0.001
    lr = 0.01
    lr = 0.1

实验结果：

    lr=0.001
    final train MSE = 0.968398
    final val MSE   = 0.986642
    final b         = 1.166612

    lr=0.01
    final train MSE = 0.245314
    final val MSE   = 0.250859
    final b         = 2.016874

    lr=0.1
    final train MSE = 0.245314
    final val MSE   = 0.250858
    final b         = 2.016968

结论：

- lr=0.001 的更新步长较小，在 500 epoch 内尚未充分收敛。
- lr=0.01 可以稳定收敛。
- lr=0.1 在当前问题上收敛更快，并没有发生发散。
- 学习率是否过大不能只根据数值判断，需要结合实际 loss 曲线和训练稳定性判断。

生成学习率对比图：

    python compare_lr.py

结果：

    outputs/lr_comparison.png


## 8. E3：失败实验与单变量修复

### 8.1 失败现象

保持以下条件不变：

    seed = 0
    data_seed = 42
    epochs = 500
    device = cpu

将 learning rate 设置为：

    lr = 0.2

失败实验第一轮结果：

    initial w = 1.540996
    initial b = -0.293429

    epoch=0
    train_loss = 104.568260
    val_loss   = 101.214691
    w          = 6.4783
    b          = 0.6207

随后梯度变为非有限数值，并触发：

    assert torch.isfinite(w.grad).all()

程序最终产生：

    AssertionError

失败日志保存在：

    outputs/e3_fail_lr02/failure.log

### 8.2 原因假设

假设失败原因是 learning rate 过大。

较大的更新步长使参数一次更新就越过较优区域，随后参数在最优点两侧产生越来越大的震荡，最终使 loss 和 gradient 数值失稳。

### 8.3 单变量修复

只修改一个因素：

    lr: 0.2 -> 0.1

其他条件保持不变。

修复实验：

    final w = 3.003685
    final b = 2.016968

    val MSE  = 0.250858
    test MSE = 0.254345

    baseline test MSE = 75.838669
    reload max abs diff = 0.0

模型重新稳定收敛。

因此该实验支持：

    learning rate 过大
        ->
    参数更新步长过大
        ->
    optimization instability
        ->
    非有限梯度

这一原因假设。


## 9. Checkpoint 与独立评估

训练结束后保存：

    model.pt

checkpoint 中包含：

    w
    b
    fixed_x
    fixed_pred_before_save

eval.py 使用：

    torch.load(..., map_location="cpu")

重新加载 checkpoint。

在固定 CPU 输入上的预测一致性检查结果：

    reload max abs diff = 0.0

满足：

    max abs prediction difference <= 1e-6

此外，训练过程中检查：

    w.grad is not None
    b.grad is not None

以及：

    torch.isfinite(w.grad).all()
    torch.isfinite(b.grad).all()

用于及时发现梯度缺失、NaN 或 Inf。


## 10. K1：为什么 zero_grad 要放在 backward 和 step 之前？

PyTorch 默认会累积梯度。

一次标准参数更新过程为：

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

其中：

    zero_grad()
        清除上一轮保存在 parameter.grad 中的梯度

    backward()
        根据当前参数和当前 loss 计算新的梯度

    step()
        根据当前梯度更新参数

如果忘记 zero_grad，新的梯度会和前一轮梯度累加。

因此参数更新使用的将不再只是当前训练步骤的梯度，可能导致错误的更新方向或异常大的更新幅度。

核心过程：

    current parameters
        ->
    forward
        ->
    loss
        ->
    zero old gradients
        ->
    backward
        ->
    new gradients
        ->
    optimizer.step()
        ->
    updated parameters


## 11. K2：train / validation / test 分别有什么作用？

Train set：

    用于计算 loss、gradient，并直接更新模型参数。

Validation set：

    不直接用于参数更新。

    用于学习率、模型结构、训练轮数等超参数选择，以及观察模型是否发生过拟合。

Test set：

    只用于最终独立性能报告。

    不应该参与模型训练或超参数选择。

正确关系：

    train
      ->
    parameter learning

    validation
      ->
    model / hyperparameter selection

    test
      ->
    final unbiased evaluation

如果需要数据标准化，例如：

    x_normalized = (x - mean) / std

其中 mean 和 std 也只能从 train set 计算，再应用到 validation 和 test。

否则同样会造成数据泄漏。


## 12. K3：相同 seed 是否保证 GPU bitwise identical？

不能保证。

固定 random seed 主要控制随机状态，例如：

    parameter initialization
    data shuffling
    random sampling

但是 GPU 计算还可能受到以下因素影响：

- 非确定性 CUDA / cuDNN 算子
- GPU 并行执行与线程调度顺序
- 浮点数运算顺序
- PyTorch / CUDA / driver 版本
- 不同 GPU 硬件
- 不同运行环境

因此：

    same seed

并不严格等价于：

    bitwise identical result

seed 的主要作用是提高实验可复现性，而不是保证所有硬件和软件环境下每一位结果都完全一致。


## 13. 从新终端复现实验

进入项目：

    cd ~/embodied_learning/week01

激活虚拟环境：

    source .venv/bin/activate

检查 Python：

    which python
    python --version

训练：

    python train.py \
      --seed 0 \
      --lr 0.01 \
      --device cpu \
      --output-dir outputs/reproduce_seed0

独立评估：

    python eval.py \
      --run-dir outputs/reproduce_seed0

成功后应生成：

    outputs/reproduce_seed0/
    ├── config.json
    ├── metrics.json
    ├── history.json
    ├── loss_curve.png
    ├── model.pt
    └── eval_metrics.json

主要验收条件：

    test MSE < train-mean baseline MSE

    reload max abs diff <= 1e-6

    gradients are not None

    gradients are finite

