import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.tensorboard import SummaryWriter

train_data=np.array(train_data,dtype=np.float32)

x=torch.tensor(train_data[:,:2])
y=torch.tensor(train_data[:,2:3])
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 将1维的数据转换为2维数据
# y = x.pow(2) + 0.2 * torch.rand(x.size())

# 将tensor置入Variable中
x, y = Variable(x).to(device), Variable(y).to(device)


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 定义一个构建神经网络的类
class Net(torch.nn.Module):  # 继承torch.nn.Module类
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 获得Net类的超类（父类）的构造方法
        # 定义神经网络的每层结构形式
        # 各个层的信息都是Net类对象的属性
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self, x):
        x = F.relu(self.hidden(x))  # 对隐藏层的输出进行relu激活
        x = self.predict(x)
        return x

    # 定义神经网络


net = Net(2, 512, 1).to(device)
print(net)  # 打印输出net的结构

# 定义优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入网络参数和学习率
loss_function = torch.nn.MSELoss()  # 最小均方误差

# 神经网络训练过程
plt.ion()  # 动态学习过程展示
plt.show()

for t in tqdm(range(3000)):
    prediction = net(x.to(device))  # 把数据x喂给net，输出预测值
    loss = loss_function(prediction, y)  # 计算两者的误差，要注意两个参数的顺序
    optimizer.zero_grad()  # 清空上一步的更新参数值
    loss.backward()  # 误差反相传播，计算新的更新参数值
    optimizer.step()  # 将计算得到的更新值赋给net.parameters()
    if t % 100 == 0:
        print(' | epoch: ', epoch, '| train loss: ', loss.data.cpu().numpy())

    # 可视化训练过程
# plt.scatter(x.data.numpy(), y.data.numpy())
p=prediction.cpu()
yy=y.cpu()
plt.cla()
plt.scatter(yy.data.numpy(), p.data.numpy())
