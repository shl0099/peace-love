import numpy as np
# a1 = [1,2,3,4]
#
# a2 = np.array([1,2,3,4])
#
# print(a2)
# print(type(a2))
# # <class 'numpy.ndarray'>   ndarray 专门进行数值计算的数组
#
# print(a2.shape)
# # (4,) 这是一个数组 里面有一个元素
#
# x = np.array([
#     [1,2],
#     [3,4],
#     [5,6]
# ])
# print(x.shape)
# # (3, 2) 三行两列
#
# # 以后看到x.shape == (500,) 说明有五百个数
# #        x.shape == (500, 1) 说明有五百个样本 每个样本有一个特征值
# # x = np.random.randn(500)  第一反应是x的shape是（500，）
#
# x1 = np.array([10,20,30,40,50])
# print(x1[0])
# # 取第一个
# print(x1[:3])
# # 取前三个
# print(x1[1:4])
# # 取2-4个
#
# '''
# 这个东西之后马上会用于划分训练集、验证集、测试集。
# x_train = x[:350]
# x_val = x[350:425]
# x_test = x[425:]
# 表明
# 训练：70%
# 验证：15%
# 测试：15%
# 也就是
# 350
# 75
# 75
# 共500个样本
# '''
#
# print(np.mean(x))
# # 求平均值
# print(np.sum(x))
#
# # 平方差
# # mse = np.mean((y_true - y_pred) ** 2)
#
# # * 和 @ 的区别
# A = np.array(
#     [[1,2],
#     [3,4]]
# )
#
# B = np.array(
#     [[5,6],
#     [7,8]]
# )
# A*B 是逐元素乘法  A@B  是矩阵乘法
# 1. (5,)
# 2. 取1-3的值 [1,2,3]
# 3. y[3,5,7,9,11]
# 4.  3
# 5. （100，5）
# 6. A*B逐元素相乘 A@B AB两个矩阵相乘

# y = 3x+2+c
# 3是权重 2是真实偏执 c是噪声
# np.random.seed(42)
# # 固定随机种子 让这次生成的随机数据尽量可以重复
# x = np.random.uniform(-5, 5, 500)
# # 生成500个随机数 每个数大致均匀分布在-5到5之间
# noise = np.random.normal(0, 0.5, 500)
# # 这行代码是用 Python 的 NumPy 库生成高斯噪声（正态分布随机数）的标准写法
# # 0.5表示标准差 大部分数值会落在-0.5 -- 0.5之间
# y = 3 * x + 2 + noise
# 1. (42,)
# 2. (500,)
# 3. (500,)
# 4. 8
# 5. 更符合真实实验数据
# 6. 固定随机种子 确保这次生成的随机数据尽量可重复

# [0,1,2,3,4,5,6]
# [6,7]
# [8,9]
# 4个
# 为了保证x和y是对应的
# 350 75 75
"""
训练集 train
→ 用来学习模型参数

验证集 validation
→ 用来选超参数、比较方案

测试集 test
→ 最后评价模型真实表现
"""

'''
1.训练集
2.测试集w和b的选择
3.test mse
4.保证公平评价
5.我觉得在做训练集的时候太在意训练集的mse导致参数是为了训练集而调的
'''

"""
训练集：学 w、b
验证集：选 learning rate、迭代次数这类超参数
测试集：最后做一次相对公平的最终评估

train：更新 w, b
val：比较学习率，选方案
test：最终评估

"""
# w = 0.0
# b = 0.0
# lr = 0.01

# 核心代码
# for epoch in range(1000):
#
#     y_pred = w * x_train + b
#
#     loss = np.mean((y_pred - y_train) ** 2)
#
#     error = y_pred - y_train
#
#     dw = 2 * np.mean(error * x_train)
#     db = 2 * np.mean(error)
#
#     w -= lr * dw
#     b -= lr * db

# 1.7
# 2.-2
# 3.2
# 4.损失度是预测值减去训练值的平方后除以总样本数
# 5.因为我们要找到最接近真实w的值,w += lr*dw可能越来越发散
# 6.lr很大，w的值趋于无穷，得不到结果

# 1.w*x_train+b
# 2.y_pred - y_train
# 3.np.mean(error**2)
# 4.dw = 2*np.mean(error * x_train)
# 5.db = 2*np.mean(error)
# 6.w = w - lr*dw
# 7.b = b - lr*db

# x_val = x[350,425]
# y_val = y[350,425]
# x_test = x[425,]
# y_test = y[425,]
# 1.train数据
# 2.x_test = x[425,] y_test = y[425,]

# 1.x[5]-x[9]
# 2.x[15]-x[N-1]
# 3.y_test_pred = w*x_test + b   test_mse = np.mean((y_test_pred-y_test)**2)

# 1.因为对不同的lr 最后训练出的w和b的值可能是不一样的
# 2.因为要保证在验证时的均值方差也很小说明训练是有效果的

# 1.w*x_val + b
# np.mean((y_val_pred-y_val)**2)