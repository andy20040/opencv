import torch
import torch.nn as nn

# ==========================================
# Question 1: LeNet-5
# ==========================================
class LeNet5(nn.Module):
    def __init__(self, activation='relu'):
        super(LeNet5, self).__init__()
        
        # C1: 卷積層
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        
        # S2: 池化層 (定義為物件，才會顯示在 summary 中) 
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C3: 卷積層
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # S4: 池化層
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C5: 卷積層 (投影片使用 Conv2d 而非 Linear) 
        # Input: 16x5x5 -> Output: 120x1x1 (因為 Kernel=5, 5-5+1=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        
        # F6: 全連接層
        self.fc1 = nn.Linear(120, 84)
        
        # Output: 輸出層
        self.fc2 = nn.Linear(84, 10)

        # 激活函數
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU()
            
        # Flatten 用
        self.flatten = nn.Flatten()

    def forward(self, x):
        # C1 -> Act -> S2
        x = self.pool1(self.act(self.conv1(x)))
        
        # C3 -> Act -> S4
        x = self.pool2(self.act(self.conv2(x)))
        
        # C5 -> Act
        x = self.act(self.conv3(x))
        
        # Flatten (變成 120)
        x = self.flatten(x)
        
        # F6 -> Act
        x = self.act(self.fc1(x))
        
        # Output
        x = self.fc2(x)
        return x

