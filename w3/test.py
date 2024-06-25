import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 检查是否可以使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 定义Res2Net残差块
class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scales=4, base_width=26):
        super(Res2NetBlock, self).__init__()
        self.scales = scales
        width = out_channels // scales

        self.conv1 = nn.Conv2d(in_channels, width * scales, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scales)

        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False) if i == 0
            else nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
            for i in range(scales - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scales - 1)])

        self.conv3 = nn.Conv2d(width * scales, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, out.size(1) // self.scales, 1)
        out = spx[0]
        for i in range(1, self.scales):
            if i == 1:
                sp = self.relu(self.bns[i-1](self.convs[i-1](spx[i])))
            else:
                sp = sp + spx[i]
                sp = self.relu(self.bns[i-1](self.convs[i-1](sp)))
            out = torch.cat((out, sp), 1)
        
        out = torch.cat((out, spx[-1]), 1)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 定义Res2Net
class Res2Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Res2Net, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def Res2Net18():
    return Res2Net(Res2NetBlock, [2, 2, 2, 2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Res2Net18().to(device)
print(net)

# 数据加载和预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络并记录训练和验证损失和准确性
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(10):  # 训练多个epoch

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 验证阶段
    net.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(testloader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')

print('Finished Training')

# 绘制损失和准确性曲线
epochs = range(1, 11)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
