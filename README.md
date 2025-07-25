# 手工实现神经网络模型集合 (Coding Models by Hands)

本项目使用PyTorch从零开始实现了多种经典的神经网络模型，包括线性回归、逻辑回归和Transformer模型。项目旨在帮助理解神经网络的底层原理和实现细节。

## 📋 项目内容

### 1. 线性回归模型 (Linear Regression)
**文件**: `linear_regression.py`

实现了两种线性回归模型：
- **简单线性方程**: y = ax + b （单变量线性回归）
- **多元线性回归**: y = w₁x₁ + w₂x₂ + w₃x₃ + b （多变量线性回归）

**特点**:
- 使用单层神经网络实现
- 适用于连续数值预测
- 模型简单，训练快速
- 具有良好的可解释性

**应用场景**: 房价预测、销量预测、股价预测等

### 2. 逻辑回归模型 (Logistic Regression)
**文件**: `logistic_regression.py`

实现了基于神经网络的二分类逻辑回归模型：
- 使用Sigmoid激活函数
- 输出类别概率
- 包含数据生成、训练、评估和可视化功能

**特点**:
- 适用于二分类任务
- 输出概率值，便于理解
- 包含完整的训练和评估流程
- 支持结果可视化

**应用场景**: 垃圾邮件检测、医疗诊断、市场分析等

### 3. 简单Transformer模型 (Simple Transformer)
**文件**: `simple_transformer.py`

从零实现了一个完整的Transformer模型，包含：
- **多头自注意力机制** (Multi-Head Self-Attention)
- **位置编码** (Positional Encoding)
- **前馈神经网络** (Feed-Forward Network)
- **层归一化** (Layer Normalization)
- **残差连接** (Residual Connections)

**核心组件**:
- `MultiHeadAttention`: 多头注意力机制
- `FeedForward`: 前馈网络
- `TransformerBlock`: Transformer块
- `SimpleTransformer`: 完整的Transformer模型

**特点**:
- 支持序列到序列的建模
- 包含注意力机制的可视化
- 适用于文本生成和序列预测
- 支持自回归生成

**应用场景**: 机器翻译、文本生成、语音识别等

## 🚀 使用方法

### 快速开始
运行完整的使用示例：
```bash
python usage_examples.py
```

### 单独运行各模型

#### 线性回归
```bash
python linear_regression.py
```

#### 逻辑回归
```bash
python logistic_regression.py
```

#### Transformer模型
```bash
python simple_transformer.py
```

## 📊 模型比较

| 模型 | 任务类型 | 输入类型 | 输出类型 | 复杂度 | 训练速度 |
|------|----------|----------|----------|--------|----------|
| 线性回归 | 回归 | 数值特征 | 连续数值 | 低 | 快 |
| 逻辑回归 | 分类 | 数值特征 | 类别概率 | 低 | 快 |
| Transformer | 序列建模 | 序列数据 | 序列/分类 | 高 | 慢 |

## 📚 代码结构

```
├── linear_regression.py      # 线性回归模型实现
├── logistic_regression.py    # 逻辑回归模型实现
├── simple_transformer.py     # Transformer模型实现
├── usage_examples.py         # 完整使用示例
└── README.md                # 项目说明文档
```

## 🔧 依赖环境

```python
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0  # 用于逻辑回归可视化
```

安装依赖：
```bash
pip install torch numpy matplotlib
```

## 💡 学习重点

### 线性回归
- 理解神经网络如何拟合线性关系
- 学习梯度下降优化过程
- 掌握损失函数的设计

### 逻辑回归
- 理解分类问题的神经网络解法
- 学习Sigmoid函数的作用
- 掌握二分类的评估方法

### Transformer
- 理解注意力机制的原理
- 学习位置编码的重要性
- 掌握序列建模的方法
- 理解残差连接和层归一化的作用

## 🎯 项目特色

1. **从零实现**: 所有模型都是使用PyTorch基础组件从零构建
2. **详细注释**: 代码包含丰富的中文注释，便于理解
3. **完整示例**: 每个模型都包含完整的训练和测试示例
4. **渐进式学习**: 从简单的线性模型到复杂的Transformer，循序渐进
5. **实用性强**: 所有示例都可以直接运行并查看结果

## 📈 扩展方向

- 添加更多激活函数的实现
- 实现卷积神经网络(CNN)
- 添加循环神经网络(RNN/LSTM)
- 实现更复杂的优化算法
- 添加正则化技术

## 📄 更新日志

- **2023/5/5**: 第一集：使用神经网络解决线性回归问题
- **最新更新**: 添加逻辑回归和Transformer模型，完善使用示例和文档

---

**作者**: [Your Name]  
**项目目标**: 通过手工实现帮助理解神经网络的基本原理  
**学习建议**: 建议按照线性回归 → 逻辑回归 → Transformer的顺序学习
