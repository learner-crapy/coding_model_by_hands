# coding_model_by_hands
第一集：使用神经网络解决线性回归问题；2023/5/5


第三集：添加 Transformer 案例；2024/8/27

## 内容概览

1. **线性回归 (linear_regression.py)**
   - 使用全连接层(Linear)实现一元/多元线性回归。
   - 运行示例会随机生成数据并训练模型。

2. **Transformer (transformer_model.py)**
   - 通过 `torch.nn.Transformer` 构建迷你 Encoder-Decoder，用于序列到序列任务。
   - 带有位置编码、掩码以及完整的前向过程示例。

## 使用方法

```bash
# 线性回归示例（训练 1000 个 epoch 并输出预测结果）
python linear_regression.py

# Transformer 示例（随机输入输出序列，打印输出张量形状）
python transformer_model.py
```

运行成功将分别看到损失变化 / 输出形状等调试信息，便于快速理解模型结构与数据流。
