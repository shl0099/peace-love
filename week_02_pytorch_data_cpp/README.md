# W02 — PyTorch MLP Training & C++17 CSV Validator

## 1. 项目目标

本周完成两条基础工程流程：

1. 使用 PyTorch 实现二维合成数据的 MLP 分类训练。
2. 使用 C++17 + CMake 实现 CSV 数据校验，并与 Python 校验器进行一致性测试。

同时验证：
- Train / Validation / Test 数据划分
- SGD 和 Adam 的多随机种子训练
- Best checkpoint 保存与独立测试
- 训练中断后的确定性 Resume
- C++ 编译、数据校验及自动化回归测试

## 2. 数据集

使用固定随机种子生成 1000 个二维样本。

- Class 0：第一维位于 [-2.0, -0.5]
- Class 1：第一维位于 [0.5, 2.0]
- Data seed：42
- Split seed：2026
- Train / Val / Test：700 / 150 / 150

两个类别在第一维上存在明显间隔，因此数据是线性可分的。

## 3. 模型结构

输入维度：2

Linear(2, 8) → ReLU → Linear(8, 2)

- 损失函数：CrossEntropyLoss
- 参数量：42
- Batch size：32
- Epochs：10
- 训练设备：CPU


## 4. 运行环境

本项目在 WSL2 Ubuntu 上开发和测试。

- Python：3.14.4
- C++ 编译器：GNU g++ 15.2.0
- CMake：4.2.3
- GNU Make：4.4.1
- 训练设备：CPU

Python 依赖版本将在 environment.lock 中记录。

## 5. 训练与测试

### 5.1 训练配置

- SGD 学习率：0.1
- Adam 学习率：0.01
- Train seeds：0、1、2
- Data seed：42（固定）
- Split seed：2026（固定）
- 每次训练 10 epochs

### 5.2 训练命令

SGD：

    python train.py --optimizer sgd --train-seed 0
    python train.py --optimizer sgd --train-seed 1
    python train.py --optimizer sgd --train-seed 2

Adam：

    python train.py --optimizer adam --train-seed 0
    python train.py --optimizer adam --train-seed 1
    python train.py --optimizer adam --train-seed 2

### 5.3 独立测试

训练结束后，使用验证集 loss 最低的 checkpoint 进行测试。

例如：

    python eval.py --optimizer sgd --train-seed 0
    python eval.py --optimizer adam --train-seed 0

其他随机种子只需要修改 train-seed 参数。

正式测试集包含 150 个样本。


## 6. Resume 确定性验证

运行命令：

    python resume_check.py

本实验使用 Adam，在 CPU 上比较两条训练路径：

- Uninterrupted：连续训练 10 epochs。
- Resumed：训练 5 epochs，保存完整 checkpoint，重新创建模型与优化器并恢复状态，再训练 5 epochs。

Checkpoint 保存以下状态：

- model_state：模型参数
- optimizer_state：Adam 内部状态
- epoch：训练进度
- torch_rng_state：PyTorch 随机状态
- train_loader_rng_state：DataLoader 随机状态

### 6.1 正例实验

完整恢复所有状态：

    max prediction diff = 0.0
    pass <= 1e-5 = True

结果：PASS。

### 6.2 反例实验

不恢复 optimizer_state：

    max prediction diff = 3.9079566

不恢复 train_loader_rng_state：

    max prediction diff = 0.0091872

两组实验都超过 1e-5 阈值，结果为 FAIL。

### 6.3 结论

仅恢复模型参数不足以保证训练轨迹一致。

对于当前按 epoch 边界恢复的 CPU 实验，还需要恢复优化器状态和随机数生成器状态。

本实验在同一个 Python 脚本中模拟训练中断和恢复，不等同于已经验证独立进程重启后的全部场景。


## 7. C++17 CSV Validator

### 7.1 编译

使用 CMake 构建，编译标准为 C++17。

编译时开启 -Wall、-Wextra、-Wpedantic 警告选项。

    cmake -S cpp -B cpp/build
    cmake --build cpp/build

### 7.2 运行

C++ 校验器：

    ./cpp/build/csv_validator data/good.csv

Python 校验器：

    python python_validator.py data/good.csv

程序返回 0 表示校验通过，返回非零值表示失败。

### 7.3 数据校验规则

- 每行必须恰好包含 3 列。
- 不允许空行。
- 数值必须能够完整解析。
- 不允许 NaN、inf 等非有限数值。
- 非法数值后缀和多余的末尾逗号必须被拒绝。

### 7.4 自动化回归测试

运行：

    bash checks/test_csv.sh

测试覆盖合法数据、错误列数、空行、NaN、inf、非法数字后缀和末尾多余逗号。

本次测试结果：

    Passed: 7
    Failed: 0
    test_exit_code=0

测试证据保存在 results/csv_validation.log。

当前程序用于约定格式的简单数值 CSV 校验，不是通用 RFC 4180 CSV 解析器。
