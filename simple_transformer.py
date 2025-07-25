'''
简单的Transformer模型实现
包含自注意力机制和前馈网络
用于序列到序列的任务，如简单的文本生成
'''

import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换和分头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最终线性变换
        output = self.W_o(attention_output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_length, :].to(x.device)
        x = self.dropout(x)
        
        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        x = self.layer_norm(x)
        output = self.output_projection(x)
        
        return output


def create_padding_mask(seq, pad_token=0):
    """创建填充掩码"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """创建前瞻掩码，用于解码器"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def generate_simple_sequence_data(vocab_size=1000, seq_length=10, num_samples=1000):
    """生成简单的序列数据用于训练"""
    # 生成随机序列，目标是预测下一个token
    torch.manual_seed(42)
    data = torch.randint(1, vocab_size, (num_samples, seq_length))
    
    # 输入是序列的前n-1个token，目标是第2到n个token
    inputs = data[:, :-1]
    targets = data[:, 1:]
    
    return inputs, targets


def train_simple_transformer():
    """训练简单的Transformer模型"""
    # 参数设置
    vocab_size = 1000
    d_model = 128
    n_heads = 8
    n_layers = 4
    d_ff = 512
    max_seq_length = 50
    seq_length = 10
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # 生成数据
    inputs, targets = generate_simple_sequence_data(vocab_size, seq_length, 1000)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = SimpleTransformer(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("开始训练Transformer模型...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            # 创建掩码
            mask = create_look_ahead_mask(batch_inputs.size(1))
            
            # 前向传播
            outputs = model(batch_inputs, mask)
            loss = criterion(outputs.reshape(-1, vocab_size), batch_targets.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return model


def generate_text(model, start_sequence, max_length=20, vocab_size=1000):
    """使用训练好的模型生成文本序列"""
    model.eval()
    
    with torch.no_grad():
        current_sequence = start_sequence.clone()
        
        for _ in range(max_length - len(start_sequence)):
            # 创建掩码
            mask = create_look_ahead_mask(current_sequence.size(1))
            
            # 预测下一个token
            outputs = model(current_sequence.unsqueeze(0), mask)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # 添加到序列中
            current_sequence = torch.cat([
                current_sequence, 
                torch.tensor([next_token])
            ])
            
            # 如果生成了结束token，停止生成
            if next_token == 0:  # 假设0是结束token
                break
    
    return current_sequence


if __name__ == "__main__":
    print("训练简单的Transformer模型...")
    model = train_simple_transformer()
    
    print("\n使用训练好的模型生成序列:")
    
    # 生成一些示例序列
    start_sequences = [
        torch.tensor([1, 2, 3]),
        torch.tensor([5, 10, 15]),
        torch.tensor([100, 200, 300])
    ]
    
    for i, start_seq in enumerate(start_sequences):
        generated = generate_text(model, start_seq, max_length=15)
        print(f"起始序列 {i+1}: {start_seq.tolist()}")
        print(f"生成序列: {generated.tolist()}")
        print()
    
    # 测试模型的注意力机制
    print("测试注意力机制:")
    test_input = torch.randint(1, 1000, (1, 8))  # batch_size=1, seq_length=8
    mask = create_look_ahead_mask(8)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input, mask)
        print(f"输入序列: {test_input[0].tolist()}")
        print(f"输出形状: {output.shape}")
        print(f"预测的下一个token概率分布的形状: {output[0, -1, :].shape}")