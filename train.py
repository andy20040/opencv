import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import LeNet5, ResNet18_Modified

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 建立儲存資料夾
if not os.path.exists('./model'):
    os.makedirs('./model')

# ==========================================
# 訓練函式
# ==========================================
def train_model(model, train_loader, val_loader, epochs, lr, save_name, model_save_path):
    criterion = nn.CrossEntropyLoss() # [cite: 89]
    optimizer = optim.Adam(model.parameters(), lr=lr) # [cite: 90]

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []
    
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 儲存最佳模型 [cite: 138]
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

    # 繪製並儲存圖表 [cite: 139]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Valid Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_hist, label='Train Acc')
    plt.plot(val_acc_hist, label='Valid Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.savefig(f"./model/{save_name}.jpg")
    plt.close()
    print(f"Training {save_name} finished.")

# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    # ---------------------------
    # Q1: MNIST LeNet-5
    # ---------------------------
    # Transform: Resize (28->32), Normalize [cite: 86, 126]
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # 1. Train LeNet with ReLU
    print("Training LeNet-5 (ReLU)...")
    model_relu = LeNet5(activation='relu').to(device)
    train_model(model_relu, train_loader, val_loader, epochs=20, lr=0.001, 
                save_name="Loss&Acc_Relu", model_save_path="./model/Weight_Relu.pth")
    
    # 2. Train LeNet with Sigmoid
    print("Training LeNet-5 (Sigmoid)...")
    model_sigmoid = LeNet5(activation='sigmoid').to(device)
    train_model(model_sigmoid, train_loader, val_loader, epochs=20, lr=0.001, 
                save_name="Loss&Acc_Sigmoid", model_save_path="./model/Weight_Sigmoid.pth")

    # ---------------------------
    # Q2: CIFAR-10 ResNet18
    # ---------------------------
    # Transform [cite: 203]
    transform_cifar = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Data Augmentation [cite: 137]
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Split train/val [cite: 189]
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader_cifar = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader_cifar = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print("Training ResNet18...")
    model_resnet = ResNet18_Modified(num_classes=10).to(device)
    train_model(model_resnet, train_loader_cifar, val_loader_cifar, epochs=30, lr=0.001, 
                save_name="Loss&Acc_ResNet", model_save_path="./model/weight_resnet.pth")