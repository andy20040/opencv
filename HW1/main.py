import sys
import os
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
from model import LeNet5

class MainWindowQ1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q1 - LeNet-5 (MNIST)")
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
        
        left_layout.addWidget(self.group_q1)
        
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

        # 變數初始化
        self.q1_image_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Models
        self.lenet_path = "./HW1/model/Weight_Relu.pth"

    # ================= Q1 Functions =================
    def load_image_q1(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.png)")
        if fname:
            self.q1_image_path = fname
            self.show_image(fname, grayscale=True)

    def show_lenet_structure(self):
        model = LeNet5().to(self.device)
        print("-" * 30)
        print("LeNet-5 Architecture:")
        # Input: (1, 32, 32)
        summary(model, (1, 32, 32))
        print("-" * 30)

    def show_lenet_acc_loss(self):
        # 顯示已儲存的圖片 - ReLU 和 Sigmoid
        img_path_relu = "./HW1/Loss&Acc_Relu.jpg"
        img_path_sigmoid = "./HW1/Loss&Acc_Sigmoid.jpg"
        
        try:
            # 讀取 Sigmoid 圖片
            pil_img_sigmoid = Image.open(img_path_sigmoid)
            img_sigmoid = np.array(pil_img_sigmoid.convert('RGB'))
            img_sigmoid = cv2.cvtColor(img_sigmoid, cv2.COLOR_RGB2BGR)
            
            # 讀取 ReLU 圖片
            pil_img_relu = Image.open(img_path_relu)
            img_relu = np.array(pil_img_relu.convert('RGB'))
            img_relu = cv2.cvtColor(img_relu, cv2.COLOR_RGB2BGR)
            
            # 確保兩張圖片寬度一致
            if img_sigmoid.shape[1] != img_relu.shape[1]:
                min_width = min(img_sigmoid.shape[1], img_relu.shape[1])
                h_sigmoid = int(img_sigmoid.shape[0] * min_width / img_sigmoid.shape[1])
                h_relu = int(img_relu.shape[0] * min_width / img_relu.shape[1])
                img_sigmoid = cv2.resize(img_sigmoid, (min_width, h_sigmoid))
                img_relu = cv2.resize(img_relu, (min_width, h_relu))
            
            # 添加標籤文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_sigmoid, 'Sigmoid', (20, 40), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(img_relu, 'ReLU', (20, 40), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            
            # 垂直堆疊：Sigmoid 在上，ReLU 在下
            combined_img = np.vstack([img_sigmoid, img_relu])
            
            # 顯示合併後的圖片
            cv2.imshow("LeNet Loss & Accuracy (Sigmoid vs ReLU)", combined_img)
            
        except FileNotFoundError as e:
            print(f"Loss image not found: {e}. Please train first.")
        except Exception as e:
            print(f"Error displaying images: {e}")

    def predict_lenet(self):
        if not self.q1_image_path:
            return

        # Preprocessing - 使用與訓練完全相同的 transform
        try:
            pil_img = Image.open(self.q1_image_path).convert('L')  # 灰階
            
            # 自動檢測是否需要反轉顏色
            # MNIST 是黑底白字，如果圖片是白底黑字就需要反轉
            img_array = np.array(pil_img)
            mean_val = np.mean(img_array)
            # 如果背景偏亮（白底黑字），就反轉為黑底白字
            if mean_val > 127:
                pil_img = Image.fromarray(255 - img_array)
                
        except Exception as e:
            print(f"Error loading image: {e}")
            return
        
        # 與訓練時完全相同的 transform
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 與訓練時一致
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(pil_img).unsqueeze(0).to(self.device)

        # Load Model
        model = LeNet5(activation='relu').to(self.device)
        try:
            model.load_state_dict(torch.load(self.lenet_path, map_location=self.device, weights_only=True))
        except:
            print(f"Model file not found at {self.lenet_path}")
            return
            
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            max_prob = max_prob.item()
            pred_class = pred_idx.item()
        
        self.label_prediction.setText(f"Predict: {pred_class}")
        
        # Show Histogram (顯示預測結果)
        self.plot_histogram(probs.cpu().numpy()[0], [str(i) for i in range(10)],
                           prediction=str(pred_class),
                           max_prob=max_prob)

    # ================= Helper Functions =================
    def show_image(self, path, grayscale=False):
        try:
            # 使用 PIL 讀取（支援中文路徑）
            pil_img = Image.open(path)
            if grayscale:
                pil_img = pil_img.convert('L')  # 灰階
                img = np.array(pil_img)
                # 直接 resize 到顯示尺寸，保留細節
                img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Convert back to RGB for Qt
            else:
                pil_img = pil_img.convert('RGB')
                img = np.array(pil_img)
                # 直接 resize 到顯示尺寸，保留細節
                img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return
        
        # 直接顯示，不需要再次 resize
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_img_display.setPixmap(QPixmap.fromImage(q_img))

    def plot_histogram(self, probs, labels, title="Probability of each class", prediction=None, max_prob=None):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, probs)
        
        # 標記預測的類別
        if prediction is not None:
            pred_idx = labels.index(prediction) if prediction in labels else -1
            if pred_idx >= 0:
                bars[pred_idx].set_color('green')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel("Probability", fontsize=12)
        plt.xlabel("Class", fontsize=12)
        plt.ylim(0, 1.2)  # 增加上限以容納預測結果文字
        plt.xticks(rotation=0)
        
        # 顯示機率數值
        for i, v in enumerate(probs):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
        # 在圖表上方顯示預測結果
        if prediction is not None and max_prob is not None:
            result_text = f'✓ Prediction: {prediction} (信心度 {max_prob:.2%})'
            result_color = 'green'
            
            # 在圖表頂部添加預測結果文字框
            plt.text(0.5, 1.15, result_text, 
                    transform=plt.gca().transAxes,
                    fontsize=14, 
                    fontweight='bold',
                    ha='center',
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor=result_color, alpha=0.3, edgecolor=result_color, linewidth=2))
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindowQ1()
    window.show()
    sys.exit(app.exec_())

