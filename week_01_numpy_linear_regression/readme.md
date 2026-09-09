# Week 1 - NumPy Linear Regression

## 本周目标

- 学习 NumPy 基础
- 理解训练集、验证集、测试集
- 理解 MSE 与梯度下降
- 从零实现一元线性回归
- 比较不同 learning rate
- 完成实验结果可视化

## 实验结果

- Best learning rate: 0.01
- Best w: 3.0190
- Best b: 2.0195
- Test MSE: 0.2670
- Mean baseline MSE: 79.7631

## Loss Curve

![Loss Comparison](results/loss_comparison.png)

## 笔记
- python列表在进行数值计算时是对整个列表进行操作的，而numpy是对里面的值进行操作的
- 训练集学习参数 验证集选学习率 测试集测试最终的模型效果
- 设置随机种子可以固定很多随机过程，例如数据打乱、参数初始化和随机采样，但不能保证所有 GPU 实验完全一致。原因是 GPU 中部分并行计算和底层算子可能具有非确定性，同一组输入在不同执行顺序或不同算法实现下可能产生微小数值差异；多线程、多进程环境下如果随机数流管理不当，也可能产生不同结果。因此除了设置随机种子，还需要控制数据、环境版本，并尽量启用确定性算法
- 