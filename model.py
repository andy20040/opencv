import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# Question 1: LeNet-5
# ==========================================
class LeNet5(nn.Module):
    def __init__(self, activation='relu'):
        super(LeNet5, self).__init__()
        # Input: 1 channel (Gray), Output: 6 feature maps, Kernel: 5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) 
        # Input: 6, Output: 16, Kernel: 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully Connected Layers
        # 16 channels * 5 * 5 image size = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 classes

        # 設定激活函數 [cite: 58]
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        # C1 -> S2 (AvgPooling) [cite: 115]
        x = F.avg_pool2d(self.activation(self.conv1(x)), 2)
        # C3 -> S4 (AvgPooling)
        x = F.avg_pool2d(self.activation(self.conv2(x)), 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # C5 -> F6 -> Output
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# Question 2: ResNet18 (Modified)
# ==========================================
class ResNet18_Modified(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_Modified, self).__init__()
        # 載入預訓練或原始 ResNet18 架構
        self.resnet = models.resnet18(pretrained=False)
        
        # 修改第一層 Conv: Kernel 7x7 stride 2 -> Kernel 3x3 stride 1 [cite: 213]
        # CIFAR-10 是 3 channels
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除 Max Pooling (使用 Identity 直接通過) [cite: 213]
        self.resnet.maxpool = nn.Identity()
        
        # 修改最後一層 FC: 1000 -> 10 [cite: 214, 219]
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)