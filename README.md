# coding_model_by_hands
第一集：使用神经网络解决线性回归问题；2023/5/5

第二集：添加了一个简单的Transformer模型示例；2024/xx/xx

## Models

- Linear Regression: 使用神经网络解决线性回归问题。
- Simple Transformer: 一个基于PyTorch nn.Transformer的最小示例模型。

## Usage Examples

### Linear Regression
见 `linear_regression.py` 文件中的 Linear_Regression 类和相关训练代码。

### Simple Transformer
```python
from linear_regression import SimpleTransformer
import torch

src = torch.rand(10, 5, 3)  # (batch, seq_len, input_dim)
tgt = torch.rand(10, 5, 3)
model = SimpleTransformer(input_dim=3)
output = model(src, tgt)
print('Transformer output shape:', output.shape)
```

---

# English

## Models
- Linear Regression: Neural network for linear regression.
- Simple Transformer: Minimal example model based on PyTorch nn.Transformer.

## Usage Examples
See `linear_regression.py` for both models. Example for SimpleTransformer:

```python
from linear_regression import SimpleTransformer
import torch

src = torch.rand(10, 5, 3)
tgt = torch.rand(10, 5, 3)
model = SimpleTransformer(input_dim=3)
output = model(src, tgt)
print('Transformer output shape:', output.shape)
```
