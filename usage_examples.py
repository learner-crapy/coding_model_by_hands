'''
使用示例：展示如何使用线性回归模型和Transformer模型
包含数据准备、模型训练、预测等完整流程
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from linear_regression import linear_equations, Linear_Regression
from transformer import Transformer, TransformerClassifier


# ============= 线性回归模型使用示例 =============

def linear_regression_example():
    """线性回归模型的完整使用示例"""
    print("=== 线性回归模型示例 ===")
    
    # 1. 准备数据：y = 2x + 3 + 噪声
    torch.manual_seed(42)
    X = torch.randn(1000, 1) * 10  # 生成1000个样本
    y = 2 * X + 3 + torch.randn(1000, 1) * 0.5  # 添加噪声
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. 创建模型
    model = linear_equations()
    
    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 4. 训练模型
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # 前向传播
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # 5. 测试模型
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor([[5.0], [10.0], [15.0]])
        predictions = model(test_x)
        print(f"\n测试预测结果:")
        for i, (x_val, pred) in enumerate(zip(test_x, predictions)):
            true_y = 2 * x_val.item() + 3
            print(f"x={x_val.item():.1f}, 预测值={pred.item():.2f}, 真实值={true_y:.2f}")
    
    # 获取学习到的参数
    for name, param in model.named_parameters():
        print(f"\n学习到的参数 {name}: {param.data}")


def multi_feature_regression_example():
    """多特征线性回归示例"""
    print("\n\n=== 多特征线性回归示例 ===")
    
    # 1. 准备数据：y = 1*x1 + 2*x2 + 3*x3 + 1
    torch.manual_seed(42)
    X = torch.randn(500, 3)  # 500个样本，3个特征
    true_weights = torch.tensor([1.0, 2.0, 3.0])
    y = torch.matmul(X, true_weights) + 1 + torch.randn(500) * 0.1
    
    # 2. 创建模型
    model = Linear_Regression()
    
    # 3. 训练
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    dataset = TensorDataset(X, y.unsqueeze(1))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    epochs = 200
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        predictions = model(test_x)
        print(f"\n测试结果:")
        for i, (x_val, pred) in enumerate(zip(test_x, predictions)):
            true_y = torch.dot(x_val, true_weights) + 1
            print(f"输入={x_val.tolist()}, 预测值={pred.item():.2f}, 真实值={true_y.item():.2f}")


# ============= Transformer模型使用示例 =============

def transformer_classification_example():
    """Transformer文本分类示例"""
    print("\n\n=== Transformer文本分类示例 ===")
    
    # 1. 准备简单的文本分类数据
    vocab_size = 1000
    num_classes = 4
    seq_length = 50
    batch_size = 16
    
    # 生成模拟数据
    torch.manual_seed(42)
    # 训练数据：随机生成的token序列
    train_data = torch.randint(0, vocab_size, (200, seq_length))
    # 标签：基于序列的某种模式生成（这里简单地基于第一个token）
    train_labels = train_data[:, 0] % num_classes
    
    # 创建数据加载器
    dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 创建模型
    model = TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=seq_length,
        dropout=0.1
    )
    
    # 3. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练模型
    model.train()
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            # 前向传播
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # 5. 测试模型
    model.eval()
    with torch.no_grad():
        test_data = torch.randint(0, vocab_size, (5, seq_length))
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        print(f"\n测试预测结果: {predicted.tolist()}")


def transformer_seq2seq_example():
    """Transformer序列到序列示例（简化版）"""
    print("\n\n=== Transformer序列到序列示例 ===")
    
    # 1. 模型参数
    vocab_size = 100
    d_model = 256
    num_heads = 4
    num_layers = 2
    d_ff = 1024
    
    # 2. 创建模型
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # 3. 准备简单的序列数据（例如：复制任务）
    batch_size = 8
    src_seq_len = 10
    tgt_seq_len = 10
    
    # 生成源序列
    src = torch.randint(1, vocab_size, (batch_size, src_seq_len))
    # 目标序列（这里简单地复制源序列）
    tgt_input = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long), src[:, :-1]], dim=1)
    tgt_output = src
    
    # 4. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 5. 训练几个步骤
    model.train()
    for step in range(50):
        # 生成目标掩码
        tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # 前向传播
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        
        # 计算损失
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f'Step [{step+1}/50], Loss: {loss.item():.4f}')
    
    # 6. 简单测试
    model.eval()
    with torch.no_grad():
        test_src = torch.randint(1, vocab_size, (1, 5))
        test_tgt = torch.zeros(1, 1, dtype=torch.long)
        
        print(f"\n测试输入序列: {test_src[0].tolist()}")
        
        # 简单的贪婪解码
        for i in range(5):
            tgt_mask = model.generate_square_subsequent_mask(test_tgt.size(1)).to(test_src.device)
            output = model(test_src, test_tgt, tgt_mask=tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1)
            test_tgt = torch.cat([test_tgt, next_token.unsqueeze(1)], dim=1)
        
        print(f"模型输出序列: {test_tgt[0, 1:].tolist()}")


def attention_visualization_example():
    """注意力权重可视化示例"""
    print("\n\n=== 注意力机制可视化示例 ===")
    
    # 创建一个简单的多头注意力层
    from transformer import MultiHeadAttention
    
    d_model = 64
    num_heads = 4
    seq_len = 10
    batch_size = 1
    
    attention = MultiHeadAttention(d_model, num_heads)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 获取注意力输出和权重
    output, attention_weights = attention(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"\n第一个头的注意力权重矩阵（前5x5）:")
    print(attention_weights[0, 0, :5, :5].detach().numpy())


if __name__ == "__main__":
    # 运行所有示例
    
    # 线性回归示例
    linear_regression_example()
    multi_feature_regression_example()
    
    # Transformer示例
    transformer_classification_example()
    transformer_seq2seq_example()
    attention_visualization_example()
    
    print("\n所有示例运行完成！")