'''
使用神经网络解决逻辑回归问题
模型：sigmoid(WTX + b)，用于二分类问题
'''

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def generate_classification_data(n_samples=1000):
    """生成二分类数据"""
    # 生成两个特征
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    
    # 创建线性边界: x1 + x2 > 0
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    
    # 添加一些噪声
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


def train_logistic_regression():
    """训练逻辑回归模型"""
    # 生成数据
    X, y = generate_classification_data(1000)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 创建模型
    model = LogisticRegression(input_size=2)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        print(f'测试集准确率: {accuracy.item():.4f}')
    
    return model, X_test, y_test


def visualize_results(model, X_test, y_test):
    """可视化分类结果"""
    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = (predictions > 0.5).float()
    
    # 转换为numpy数组
    X_np = X_test.numpy()
    y_np = y_test.numpy().flatten()
    pred_np = predicted_labels.numpy().flatten()
    
    # 创建图形
    plt.figure(figsize=(12, 4))
    
    # 真实标签
    plt.subplot(1, 2, 1)
    colors = ['red' if label == 0 else 'blue' for label in y_np]
    plt.scatter(X_np[:, 0], X_np[:, 1], c=colors, alpha=0.6)
    plt.title('真实标签')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    # 预测标签
    plt.subplot(1, 2, 2)
    colors = ['red' if label == 0 else 'blue' for label in pred_np]
    plt.scatter(X_np[:, 0], X_np[:, 1], c=colors, alpha=0.6)
    plt.title('预测标签')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    plt.tight_layout()
    plt.savefig('logistic_regression_results.png')
    print("结果可视化已保存为 'logistic_regression_results.png'")


if __name__ == "__main__":
    print("开始训练逻辑回归模型...")
    model, X_test, y_test = train_logistic_regression()
    
    print("\n生成可视化结果...")
    visualize_results(model, X_test, y_test)
    
    # 使用训练好的模型进行预测
    print("\n使用训练好的模型进行预测:")
    test_samples = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(test_samples)
        for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
            print(f"样本 {sample.tolist()}: 预测概率 = {pred.item():.4f}, 预测类别 = {int(pred.item() > 0.5)}")