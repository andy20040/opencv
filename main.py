import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QGroupBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image

# 匯入模型定義
from model import LeNet5, ResNet18_Modified

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw2_StudentID_Name")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左側控制面板 (按鈕)
        left_layout = QVBoxLayout()
        
        # Q1 Group
        self.group_q1 = QGroupBox("1. LeNet-5 (MNIST)")
        q1_layout = QVBoxLayout()
        self.btn_load_q1 = QPushButton("Load Image")
        self.btn_1_1 = QPushButton("1.1 Show Architecture")
        self.btn_1_2 = QPushButton("1.2 Show Acc Loss")
        self.btn_1_3 = QPushButton("1.3 Predict")
        
        q1_layout.addWidget(self.btn_load_q1)
        q1_layout.addWidget(self.btn_1_1)
        q1_layout.addWidget(self.btn_1_2)
        q1_layout.addWidget(self.btn_1_3)
        self.group_q1.setLayout(q1_layout)
        
        # Q2 Group
        self.group_q2 = QGroupBox("2. ResNet18 (CIFAR-10)")
        q2_layout = QVBoxLayout()
        self.btn_2_1 = QPushButton("2.1 Load and Show Image")
        self.btn_2_2 = QPushButton("2.2 Show Model Structure")
        self.btn_2_3 = QPushButton("2.3 Show Acc and Loss")
        self.btn_2_4 = QPushButton("2.4 Inference")
        
        q2_layout.addWidget(self.btn_2_1)
        q2_layout.addWidget(self.btn_2_2)
        q2_layout.addWidget(self.btn_2_3)
        q2_layout.addWidget(self.btn_2_4)
        self.group_q2.setLayout(q2_layout)

        left_layout.addWidget(self.group_q1)
        left_layout.addWidget(self.group_q2)
        
        # 右側顯示區域 (圖片與文字)
        right_layout = QVBoxLayout()
        self.label_img_display = QLabel("Image Display Area")
        self.label_img_display.setAlignment(Qt.AlignCenter)
        self.label_img_display.setFixedSize(400, 400)
        self.label_img_display.setStyleSheet("border: 1px solid black;")
        
        self.label_prediction = QLabel("Predict: ")
        
        right_layout.addWidget(self.label_img_display)
        right_layout.addWidget(self.label_prediction)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 連結按鈕功能
        self.btn_load_q1.clicked.connect(self.load_image_q1)
        self.btn_1_1.clicked.connect(self.show_lenet_structure)
        self.btn_1_2.clicked.connect(self.show_lenet_acc_loss)
        self.btn_1_3.clicked.connect(self.predict_lenet)

        self.btn_2_1.clicked.connect(self.load_image_q2)
        self.btn_2_2.clicked.connect(self.show_resnet_structure)
        self.btn_2_3.clicked.connect(self.show_resnet_acc_loss)
        self.btn_2_4.clicked.connect(self.predict_resnet)

        # 變數初始化
        self.q1_image_path = None
        self.q2_image_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Models (Lazy loading or Pre-loading)
        # 假設已經訓練好並存在 ./model/ 下
        self.lenet_path = "./model/Weight_Relu.pth" # [cite: 150]
        self.resnet_path = "./model/weight_resnet.pth"

    # ================= Q1 Functions =================
    def load_image_q1(self):
        # [cite: 71]
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.png)")
        if fname:
            self.q1_image_path = fname
            self.show_image(fname, grayscale=True)

    def show_lenet_structure(self):
        # [cite: 102]
        model = LeNet5().to(self.device)
        print("-" * 30)
        print("LeNet-5 Architecture:")
        # Input: (1, 32, 32) [cite: 127]
        summary(model, (1, 32, 32))
        print("-" * 30)

    def show_lenet_acc_loss(self):
        # [cite: 142] 顯示已儲存的圖片
        img_path = "./model/Loss&Acc_Relu.jpg"
        if cv2.imread(img_path) is None:
            print("Loss image not found. Please train first.")
            return
        img = cv2.imread(img_path)
        cv2.imshow("LeNet Loss & Accuracy", img)

    def predict_lenet(self):
        # [cite: 148, 153]
        if not self.q1_image_path:
            return

        # Preprocessing [cite: 125, 126]
        img = cv2.imread(self.q1_image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        # 反轉顏色 (黑底白字 -> 白底黑字 或相反，視訓練資料而定)
        # 投影片 MNIST 是黑底白字，如果輸入是白底黑字需要 bitwise_not
        # 這裡假設使用者測試圖為白底黑字，需轉為黑底白字 [cite: 161]
        img = cv2.bitwise_not(img) 
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # Load Model
        model = LeNet5(activation='relu').to(self.device)
        try:
            model.load_state_dict(torch.load(self.lenet_path, map_location=self.device))
        except:
            print(f"Model file not found at {self.lenet_path}")
            return
            
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        self.label_prediction.setText(f"Predict: {pred_class}")
        
        # Show Histogram [cite: 154]
        self.plot_histogram(probs.cpu().numpy()[0], [str(i) for i in range(10)])

    # ================= Q2 Functions =================
    def load_image_q2(self):
        # [cite: 201]
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.png)")
        if fname:
            self.q2_image_path = fname
            self.show_image(fname, grayscale=False)

    def show_resnet_structure(self):
        # [cite: 211, 222]
        model = ResNet18_Modified().to(self.device)
        print("-" * 30)
        print("ResNet18 Modified Architecture:")
        summary(model, (3, 32, 32)) # CIFAR10 input size
        print("-" * 30)

    def show_resnet_acc_loss(self):
        # [cite: 290]
        img_path = "./model/Loss&Acc_ResNet.jpg"
        if cv2.imread(img_path) is None:
            print("Loss image not found. Please train first.")
            return
        img = cv2.imread(img_path)
        cv2.imshow("ResNet Loss & Accuracy", img)

    def predict_resnet(self):
        # [cite: 307]
        if not self.q2_image_path:
            return

        # Preprocessing [cite: 203]
        img_pil = Image.open(self.q2_image_path).convert('RGB')
        img_pil = img_pil.resize((32, 32))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)

        # Load Model
        model = ResNet18_Modified(num_classes=10).to(self.device)
        try:
            model.load_state_dict(torch.load(self.resnet_path, map_location=self.device))
        except:
            print(f"Model file not found at {self.resnet_path}")
            return
            
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            max_prob = max_prob.item()
            pred_idx = pred_idx.item()

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Threshold logic for "Others" [cite: 300, 320]
        threshold = 0.5  # Adjust yourself as per slides
        if max_prob < threshold:
            pred_label = "Others"
        else:
            pred_label = classes[pred_idx]
            
        self.label_prediction.setText(f"Predict: {pred_label} ({max_prob:.2%})")
        
        # Show Histogram [cite: 302]
        self.plot_histogram(probs.cpu().numpy()[0], classes, title="Probability Distribution")

    # ================= Helper Functions =================
    def show_image(self, path, grayscale=False):
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32)) # Resize for display consistent with model input [cite: 86]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Convert back to RGB for Qt
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32)) # [cite: 203]
            
        # Resize to larger for GUI display (e.g., 400x400)
        img_display = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        h, w, ch = img_display.shape
        bytes_per_line = ch * w
        q_img = QImage(img_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_img_display.setPixmap(QPixmap.fromImage(q_img))

    def plot_histogram(self, probs, labels, title="Probability of each class"):
        plt.figure(figsize=(8, 5))
        plt.bar(labels, probs)
        plt.title(title)
        plt.ylabel("Probability")
        plt.xlabel("Class")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        for i, v in enumerate(probs):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())