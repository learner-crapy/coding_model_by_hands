'''
使用示例：展示如何使用项目中的各种神经网络模型
包含线性回归、逻辑回归和Transformer模型的使用示例
'''

import torch
import numpy as np
from linear_regression import Linear_Regression, linear_equations
from logistic_regression import LogisticRegression, train_logistic_regression, generate_classification_data
from simple_transformer import SimpleTransformer, train_simple_transformer, generate_text


def demo_linear_regression():
    """演示线性回归模型的使用"""
    print("=" * 50)
    print("线性回归模型演示")
    print("=" * 50)
    
    # 1. 简单线性方程 y = ax + b
    print("\n1. 简单线性方程演示 (y = ax + b)")
    print("-" * 30)
    
    # 生成数据
    x = torch.rand(100, 1)
    y = 2 * x + 1 + 0.1 * torch.rand(100, 1)  # y = 2x + 1 + noise
    
    # 创建模型
    model = linear_equations()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    # 训练
    print("训练中...")
    for epoch in range(1000):
        total_loss = 0
        for xi, yi in zip(x, y):
            y_pred = model(xi)
            loss = loss_fn(y_pred, yi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 200 == 0:
            avg_loss = total_loss / len(x)
            print(f"Epoch [{epoch+1}/1000], Average Loss: {avg_loss:.4f}")
    
    # 测试
    test_x = torch.tensor([[1.0], [2.0], [3.0]])
    with torch.no_grad():
        predictions = model(test_x)
        print(f"\n测试结果:")
        for i, (input_val, pred) in enumerate(zip(test_x, predictions)):
            expected = 2 * input_val + 1
            print(f"输入: {input_val.item():.1f}, 预测: {pred.item():.3f}, 期望: {expected.item():.3f}")
    
    # 2. 多元线性回归 y = w1*x1 + w2*x2 + w3*x3 + b
    print("\n\n2. 多元线性回归演示 (y = w1*x1 + w2*x2 + w3*x3 + b)")
    print("-" * 50)
    
    # 使用已有的训练代码
    X = torch.rand(100, 3, 1)
    W = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)
    
    model = Linear_Regression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    print("训练中...")
    batch_size = 10
    for epoch in range(500):
        x_batch = []
        y_batch = []
        total_loss = 0
        
        for i in range(X.shape[0]):
            x = torch.transpose(X[i], 1, 0)
            x_batch.append(x)
            y = torch.matmul(x, torch.transpose(W, 1, 0)) + 1
            y_batch.append(y)
            
            if (i + 1) % batch_size == 0:
                x_batch = torch.stack([torch.tensor(x).clone() for x in x_batch], dim=0)
                y_batch = torch.stack([torch.tensor(y).clone() for y in y_batch], dim=0)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                x_batch = []
                y_batch = []
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/500], Loss: {total_loss:.4f}")
    
    # 测试多元线性回归
    test_samples = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 1.5, 2.5]])
    with torch.no_grad():
        predictions = model(test_samples)
        print(f"\n测试结果:")
        for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
            expected = torch.matmul(sample.unsqueeze(0), torch.transpose(W, 1, 0)) + 1
            print(f"输入: {sample.tolist()}, 预测: {pred.item():.3f}, 期望: {expected.item():.3f}")


def demo_logistic_regression():
    """演示逻辑回归模型的使用"""
    print("\n\n" + "=" * 50)
    print("逻辑回归模型演示")
    print("=" * 50)
    
    # 生成分类数据
    X, y = generate_classification_data(500)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 创建模型
    model = LogisticRegression(input_size=2)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    print("训练逻辑回归模型...")
    num_epochs = 500
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
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
        print(f'\n测试集准确率: {accuracy.item():.4f}')
    
    # 展示一些预测结果
    print("\n预测示例:")
    test_samples = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [-0.5, 0.5]])
    with torch.no_grad():
        predictions = model(test_samples)
        for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
            predicted_class = int(pred.item() > 0.5)
            print(f"样本 {sample.tolist()}: 预测概率 = {pred.item():.4f}, 预测类别 = {predicted_class}")


def demo_simple_transformer():
    """演示简单Transformer模型的使用"""
    print("\n\n" + "=" * 50)
    print("简单Transformer模型演示")
    print("=" * 50)
    
    # 参数设置
    vocab_size = 100  # 减小词汇表大小以加快训练
    d_model = 64      # 减小模型维度
    n_heads = 4
    n_layers = 2
    d_ff = 128
    max_seq_length = 20
    
    # 创建模型
    model = SimpleTransformer(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length)
    
    # 生成简单的训练数据（序列模式学习）
    print("生成训练数据...")
    torch.manual_seed(42)
    
    # 创建简单的序列模式：[1, 2, 3, 4, 5] -> [2, 3, 4, 5, 6]
    sequences = []
    for i in range(200):  # 减少训练数据
        start = torch.randint(1, vocab_size - 10, (1,)).item()
        seq = torch.arange(start, start + 8)  # 长度为8的序列
        sequences.append(seq)
    
    sequences = torch.stack(sequences)
    inputs = sequences[:, :-1]  # 输入：前7个token
    targets = sequences[:, 1:]  # 目标：后7个token
    
    # 训练模型
    print("训练Transformer模型...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    num_epochs = 50  # 减少训练轮数
    batch_size = 16
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            outputs = model(batch_inputs)
            loss = criterion(outputs.reshape(-1, vocab_size), batch_targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(inputs) // batch_size)
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # 测试生成能力
    print("\n测试序列生成:")
    model.eval()
    
    test_sequences = [
        torch.tensor([10, 11, 12]),
        torch.tensor([50, 51, 52]),
        torch.tensor([5, 6, 7])
    ]
    
    from simple_transformer import create_look_ahead_mask
    
    for i, start_seq in enumerate(test_sequences):
        print(f"\n起始序列 {i+1}: {start_seq.tolist()}")
        
        current_sequence = start_seq.clone()
        
        with torch.no_grad():
            for _ in range(5):  # 生成5个新token
                mask = create_look_ahead_mask(current_sequence.size(0))
                outputs = model(current_sequence.unsqueeze(0), mask)
                next_token_logits = outputs[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                current_sequence = torch.cat([current_sequence, torch.tensor([next_token])])
        
        print(f"生成序列: {current_sequence.tolist()}")
        expected = list(range(start_seq[0].item(), start_seq[0].item() + len(current_sequence)))
        print(f"期望模式: {expected}")


def compare_models():
    """比较不同模型的特点和应用场景"""
    print("\n\n" + "=" * 50)
    print("模型比较与应用场景")
    print("=" * 50)
    
    comparisons = [
        {
            "模型": "线性回归 (Linear Regression)",
            "任务类型": "回归任务",
            "输入": "数值特征",
            "输出": "连续数值",
            "应用场景": "房价预测、销量预测、股价预测等",
            "优点": "简单、可解释性强、训练快速",
            "缺点": "只能处理线性关系"
        },
        {
            "模型": "逻辑回归 (Logistic Regression)",
            "任务类型": "分类任务",
            "输入": "数值特征",
            "输出": "类别概率",
            "应用场景": "垃圾邮件检测、医疗诊断、市场分析等",
            "优点": "输出概率、可解释性好、适合二分类",
            "缺点": "假设线性决策边界"
        },
        {
            "模型": "Transformer",
            "任务类型": "序列建模",
            "输入": "序列数据",
            "输出": "序列或分类",
            "应用场景": "机器翻译、文本生成、语音识别等",
            "优点": "处理长序列、并行计算、注意力机制",
            "缺点": "计算复杂度高、需要大量数据"
        }
    ]
    
    for comp in comparisons:
        print(f"\n{comp['模型']}:")
        for key, value in comp.items():
            if key != "模型":
                print(f"  {key}: {value}")


if __name__ == "__main__":
    print("神经网络模型使用示例")
    print("本演示将展示线性回归、逻辑回归和Transformer模型的使用方法")
    
    # 运行所有演示
    demo_linear_regression()
    demo_logistic_regression()
    demo_simple_transformer()
    compare_models()
    
    print(f"\n{'='*50}")
    print("所有演示完成！")
    print("你可以单独运行各个模型文件来进行更详细的实验。")
    print("="*50)