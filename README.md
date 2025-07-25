# coding_model_by_hands

一个从零开始手写深度学习模型的项目，包含线性回归和Transformer模型的PyTorch实现。

## 项目结构

```
.
├── README.md               # 项目说明文档
├── linear_regression.py    # 线性回归模型实现
├── transformer.py          # Transformer模型实现
└── usage_examples.py       # 使用示例代码
```

## 模型介绍

### 1. 线性回归模型

#### 简单线性回归 (linear_equations)
- **功能**：解决 y = ax + b 形式的线性方程
- **结构**：单层神经网络，包含一个线性层
- **用途**：适用于单变量线性回归问题

#### 多特征线性回归 (Linear_Regression)
- **功能**：解决 y = W^T·X + b 形式的多元线性回归
- **结构**：单层神经网络，支持多维输入
- **用途**：适用于多变量线性回归问题

### 2. Transformer模型

#### 核心组件
- **MultiHeadAttention**：多头注意力机制
  - 支持查询、键、值的灵活输入
  - 可配置的注意力头数
  - 包含dropout正则化

- **PositionalEncoding**：位置编码
  - 使用正弦和余弦函数生成位置嵌入
  - 支持最大5000长度的序列

- **EncoderLayer**：编码器层
  - 自注意力机制
  - 前馈神经网络
  - 层归一化和残差连接

- **DecoderLayer**：解码器层
  - 自注意力机制
  - 编码器-解码器交叉注意力
  - 前馈神经网络
  - 层归一化和残差连接

#### 完整模型

##### Transformer (序列到序列)
- **用途**：机器翻译、文本生成、序列转换等任务
- **特点**：
  - 完整的编码器-解码器架构
  - 支持源序列和目标序列的掩码
  - 可配置的模型维度、层数、注意力头数等

##### TransformerClassifier (文本分类)
- **用途**：文本分类、情感分析等任务
- **特点**：
  - 仅使用编码器架构
  - 自适应池化层用于序列聚合
  - 最终的分类层输出类别概率

## 使用方法

### 安装依赖

```bash
pip install torch numpy
```

### 运行示例

```bash
python usage_examples.py
```

### 快速开始

#### 线性回归示例

```python
from linear_regression import linear_equations
import torch
import torch.nn as nn

# 创建模型
model = linear_equations()

# 准备数据
X = torch.randn(100, 1)
y = 2 * X + 3 + torch.randn(100, 1) * 0.1

# 训练模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Transformer分类示例

```python
from transformer import TransformerClassifier
import torch

# 创建模型
model = TransformerClassifier(
    vocab_size=1000,
    num_classes=4,
    d_model=128,
    num_heads=4,
    num_layers=2
)

# 准备数据
input_ids = torch.randint(0, 1000, (32, 50))  # batch_size=32, seq_len=50
labels = torch.randint(0, 4, (32,))

# 前向传播
outputs = model(input_ids)
```

## 模型参数说明

### Transformer参数
- `vocab_size`: 词汇表大小
- `d_model`: 模型维度（默认512）
- `num_heads`: 注意力头数（默认8）
- `num_layers`: 编码器/解码器层数（默认6）
- `d_ff`: 前馈网络维度（默认2048）
- `max_seq_len`: 最大序列长度（默认5000）
- `dropout`: Dropout率（默认0.1）

## 特性

- ✅ 完整的PyTorch实现
- ✅ 详细的中文注释
- ✅ 模块化设计，易于理解和修改
- ✅ 包含多个使用示例
- ✅ 支持批处理
- ✅ 灵活的模型配置

## 更新日志

- 2023/5/5：第一集 - 使用神经网络解决线性回归问题
- 2024/x/x：第二集 - 添加Transformer模型实现和使用示例

## 许可证

本项目仅供学习和研究使用。
