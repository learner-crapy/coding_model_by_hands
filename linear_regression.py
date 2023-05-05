'''
使用神经网络解决线性回归问题
方程模型：y=ax+b，a和b都是一个值
两个参数，a和b,由此需要设置多少个dense层呢？
答案：两层，而且每一层只有一个神经元
'''

import torch
from torch import nn


class linear_equations(nn.Module):
    def __init__(self):
        super(linear_equations, self).__init__()
        self.a = nn.Linear(1, 1)
        self.b = nn.Linear(1, 1)

    # 输入的是一个坐标，(x,y)，x和y都是一个数值
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x


'''
线性回归
模型：y=WTX+b，W，X是向量，b, y是一个值
同样的，两层搞定,按照下面的定义，似乎偏置不能够被计算出来，以下面的例子来看，正确结果应该是13，但是偏置1没有没加上才得到12
看了手动学习深度学习的例子，可能只需要一层dense就可以，因为这里只有w和b两个参数，设置Linear(2,1)代表输入两个特征
参考https://zhuanlan.zhihu.com/p/33223290
每一个dense层都有权重和偏置，所以只需要一层足以
'''


class Linear_Regression(nn.Module):
    def __init__(self):
        super(Linear_Regression, self).__init__()
        self.W = nn.Linear(3, 1)
        self.b = nn.Linear(1, 1)

    def forward(self, x):
        x = self.W(x)
        x = self.b(x)
        return x


# 生成线性回归的数据
# 生成数据
# 假设现有线性回归模型，Y=[1,2,3]*X+1
# 生成数据, 100个数据，以Y=[1,2,3]*X+1为基础，加上一些噪声
# 随机生成shape为(3, 10)的100个数据
X = torch.rand(100, 3, 1)
# X = torch.tensor(X)
W = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)
# y = []
model = Linear_Regression()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
# 开始训练
for epoch in range(1000):
    for x in X:
        x = torch.transpose(x, 1, 0)
        y = torch.matmul(x, torch.transpose(W, 1, 0))
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

with torch.no_grad():
    print(model(torch.tensor([2.0, 2.0, 2.0])))

true_label = torch.matmul(torch.tensor([2.0, 2.0, 2.0]).unsqueeze(0), torch.transpose(W, 1, 0))+1
print(true_label)


# 模型定义完毕，下面需要进行训练，生成100个数据，以方程y=2x+1为基础，加上一些噪声
# 生成数据
x = torch.rand(100, 1)
y = 10 * x + 1 + torch.rand(100, 1)
x = sum(x.tolist(), [])
y = sum(y.tolist(), [])


# 定义优化器和损失函数
model = linear_equations()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 开始训练
for epoch in range(0, 1000):
    for xi, yi in zip(x, y):
        xi = torch.tensor(xi).unsqueeze(0)
        yi = torch.tensor(yi).unsqueeze(0)
        y_pred = model(xi)
        loss = loss_fn(y_pred, yi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        # print(list(model.parameters()))

with torch.no_grad():
    print(model(torch.tensor([2.0])))
