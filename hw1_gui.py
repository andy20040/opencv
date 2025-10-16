import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import atexit

class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing GUI")
        self.root.geometry("1400x900")
        
        # Variables
        self.image = None
        self.image_path = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Image Processing Tool", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Load image buttons
        button_frame_load = ttk.Frame(main_frame)
        button_frame_load.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        load_btn = ttk.Button(button_frame_load, text="Load Image", 
                             command=self.load_image)
        load_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 相對路徑載入按鈕
        load_relative_btn = ttk.Button(button_frame_load, text="Load rgb.jpg", 
                                      command=self.load_relative_image)
        load_relative_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # (Removed) Load image1.jpg button
        
        # Image path label
        self.path_label = ttk.Label(main_frame, text="No image loaded")
        self.path_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # 1. Image Processing section
        processing_frame = ttk.LabelFrame(main_frame, text="1. Image Processing", padding="10")
        processing_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 1.1 Color Separation
        self.separation_btn = ttk.Button(processing_frame, text="1.1 Color Separation", 
                                       command=self.show_color_separation,
                                       state='disabled')
        self.separation_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 1.2 Color Transformation
        self.transform_btn = ttk.Button(processing_frame, text="1.2 Color Transformation", 
                                      command=self.show_color_transform,
                                      state='disabled')
        self.transform_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 2. Image Smoothing section
        smoothing_frame = ttk.LabelFrame(main_frame, text="2. Image Smoothing", padding="10")
        smoothing_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 2.1 Gaussian Blur
        self.gaussian_btn = ttk.Button(smoothing_frame, text="2.1 Gaussian Blur", 
                                     command=self.show_gaussian_blur,
                                     state='disabled')
        self.gaussian_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 2.2 Bilateral Filter
        self.bilateral_btn = ttk.Button(smoothing_frame, text="2.2 Bilateral Filter", 
                                       command=self.show_bilateral_filter,
                                       state='disabled')
        self.bilateral_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 2.3 Median Filter
        self.median_btn = ttk.Button(smoothing_frame, text="2.3 Median Filter", 
                                   command=self.show_median_filter,
                                   state='disabled')
        self.median_btn.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # 3. Edge Detection section
        edge_frame = ttk.LabelFrame(main_frame, text="3. Edge Detection", padding="10")
        edge_frame.grid(row=6, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 3.1 Sobel X
        self.sobel_x_btn = ttk.Button(edge_frame, text="3.1 Sobel X", 
                                     command=self.show_sobel_x,
                                     state='disabled')
        self.sobel_x_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 3.2 Sobel Y
        self.sobel_y_btn = ttk.Button(edge_frame, text="3.2 Sobel Y", 
                                     command=self.show_sobel_y,
                                     state='disabled')
        self.sobel_y_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 3.3 Combination and Threshold
        self.combination_btn = ttk.Button(edge_frame, text="3.3 Combination & Threshold", 
                                        command=self.show_combination_threshold,
                                        state='disabled')
        self.combination_btn.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 3.4 Gradient Angle
        self.gradient_btn = ttk.Button(edge_frame, text="3.4 Gradient Angle", 
                                     command=self.show_gradient_angle,
                                     state='disabled')
        self.gradient_btn.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 4. Transforms section
        transforms_frame = ttk.LabelFrame(main_frame, text="4. Transforms", padding="10")
        transforms_frame.grid(row=7, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Create input frame for parameters
        input_frame = ttk.Frame(transforms_frame)
        input_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Rotation input
        ttk.Label(input_frame, text="Rotation:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.rotation_var = tk.StringVar(value="30")
        self.rotation_entry = ttk.Entry(input_frame, textvariable=self.rotation_var, width=10)
        self.rotation_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Scaling input
        ttk.Label(input_frame, text="Scaling:").grid(row=0, column=2, padx=(20, 5), sticky=tk.W)
        self.scaling_var = tk.StringVar(value="0.9")
        self.scaling_entry = ttk.Entry(input_frame, textvariable=self.scaling_var, width=10)
        self.scaling_entry.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        
        # Tx input
        ttk.Label(input_frame, text="Tx:").grid(row=1, column=0, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        self.tx_var = tk.StringVar(value="535")
        self.tx_entry = ttk.Entry(input_frame, textvariable=self.tx_var, width=10)
        self.tx_entry.grid(row=1, column=1, padx=5, pady=(5, 0), sticky=tk.W)
        
        # Ty input
        ttk.Label(input_frame, text="Ty:").grid(row=1, column=2, padx=(20, 5), pady=(5, 0), sticky=tk.W)
        self.ty_var = tk.StringVar(value="335")
        self.ty_entry = ttk.Entry(input_frame, textvariable=self.ty_var, width=10)
        self.ty_entry.grid(row=1, column=3, padx=5, pady=(5, 0), sticky=tk.W)
        
        # Transform button
        self.transform_btn = ttk.Button(transforms_frame, text="Transform", 
                                      command=self.apply_transform,
                                      state='disabled')
        self.transform_btn.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Display frame for matplotlib with scrollbar
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(8, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # 設定關閉事件處理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 註冊清理函數
        atexit.register(self.cleanup)
        
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            # 嘗試多種路徑方式載入圖片
            self.image_path = file_path
            
            # 方法1: 直接載入
            self.image = cv2.imread(file_path)
            
            # 方法2: 如果失敗，嘗試使用相對路徑
            if self.image is None:
                import os
                # 取得檔案名稱
                filename = os.path.basename(file_path)
                # 嘗試從當前目錄載入
                self.image = cv2.imread(filename)
                if self.image is not None:
                    self.image_path = filename
            
            # 方法3: 如果還是失敗，嘗試使用中文編碼
            if self.image is None:
                try:
                    import numpy as np
                    # 使用numpy讀取，然後轉換
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    nparr = np.frombuffer(data, np.uint8)
                    self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except:
                    pass
            
            if self.image is None:
                messagebox.showerror("Error", "Could not load the image!")
                return
            
            # Update path label
            self.path_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            
            # Enable all processing buttons
            self.separation_btn.config(state='normal')
            self.transform_btn.config(state='normal')
            self.gaussian_btn.config(state='normal')
            self.bilateral_btn.config(state='normal')
            self.median_btn.config(state='normal')
            self.sobel_x_btn.config(state='normal')
            self.sobel_y_btn.config(state='normal')
            self.combination_btn.config(state='normal')
            self.gradient_btn.config(state='normal')
            self.transform_btn.config(state='normal')
            
            messagebox.showinfo("Success", "Image loaded successfully!")
    
    def load_relative_image(self):
        """Load image using relative path (解決中文路徑問題)"""
        # 嘗試載入當前目錄下的 rgb.jpg
        relative_path = 'rgb.jpg'
        self.image_path = relative_path
        self.image = cv2.imread(relative_path)
        
        if self.image is None:
            messagebox.showerror("Error", f"Could not load {relative_path}!\n\nPlease make sure rgb.jpg is in the same folder as the program.")
            return
        
        # Update path label
        self.path_label.config(text=f"Loaded: {relative_path}")
        
        # Enable all processing buttons
        self.separation_btn.config(state='normal')
        self.transform_btn.config(state='normal')
        self.gaussian_btn.config(state='normal')
        self.bilateral_btn.config(state='normal')
        self.median_btn.config(state='normal')
        self.sobel_x_btn.config(state='normal')
        self.sobel_y_btn.config(state='normal')
        self.combination_btn.config(state='normal')
        self.gradient_btn.config(state='normal')
        self.transform_btn.config(state='normal')
        
        messagebox.showinfo("Success", f"Successfully loaded {relative_path}!")
    
    
    
    def on_closing(self):
        """處理GUI關閉事件"""
        self.cleanup()
        self.root.destroy()
        sys.exit(0)
    
    def cleanup(self):
        """清理資源"""
        try:
            # 關閉所有OpenCV窗口
            cv2.destroyAllWindows()
            
            # 關閉matplotlib圖形
            plt.close('all')
            
            # 清理顯示框架
            for widget in self.display_frame.winfo_children():
                widget.destroy()
                
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def show_color_separation(self):
        """Display RGB channel separation"""
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return
        
        # Clear previous display
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        # Process image for color separation
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        b, g, r = cv2.split(self.image)
        
        # Convert each grayscale channel back to BGR format
        r_bgr = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
        g_bgr = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
        b_bgr = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
        
        # Convert BGR back to RGB for display
        r_rgb = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2RGB)
        g_rgb = cv2.cvtColor(g_bgr, cv2.COLOR_BGR2RGB)
        b_rgb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RGB Color Separation', fontsize=18, y=0.95)
        
        # 增加子圖之間的間距，特別是垂直間距
        plt.subplots_adjust(hspace=0.6, wspace=0.3)
        
        # Display images
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original RGB Image', fontsize=14, pad=20)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(r_rgb)
        axes[0, 1].set_title('R Channel (Red)', fontsize=14, pad=20)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(g_rgb)
        axes[1, 0].set_title('G Channel (Green)', fontsize=14, pad=20)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(b_rgb)
        axes[1, 1].set_title('B Channel (Blue)', fontsize=14, pad=20)
        axes[1, 1].axis('off')
        
        # Embed matplotlib in tkinter
        canvas = FigureCanvasTkAgg(fig, self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    def show_color_transform(self):
        """Display grayscale transformations"""
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return
        
        # Clear previous display
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        # Process image for color transform
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        b, g, r = cv2.split(self.image)
        
        # Create grayscale images
        cv_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        avg_gray = (b/3 + g/3 + r/3).astype(np.uint8)
        
        # Create matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle('Color Transform - Grayscale Conversion', fontsize=18, y=0.95)
        
        # 增加子圖之間的間距
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Display images
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original RGB Image', fontsize=14, pad=20)
        axes[0].axis('off')
        
        axes[1].imshow(cv_gray, cmap='gray')
        axes[1].set_title('OpenCV Grayscale\n(cv2.cvtColor)', fontsize=14, pad=20)
        axes[1].axis('off')
        
        axes[2].imshow(avg_gray, cmap='gray')
        axes[2].set_title('Average Grayscale\n((R+G+B)/3)', fontsize=14, pad=20)
        axes[2].axis('off')
        
        # Embed matplotlib in tkinter
        canvas = FigureCanvasTkAgg(fig, self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    # Image Smoothing Functions
    def show_gaussian_blur(self):
        """Popup: OpenCV trackbar controls Gaussian kernel radius m (1..5) on currently loaded image"""
        import threading
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        def popup_worker(img):
            win_name = 'Gaussian Blur (m=1..5, kernel=(2m+1)x(2m+1))'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 900, 600)

            # Create trackbar: m in [1,5]
            cv2.createTrackbar('m', win_name, 5, 5, lambda v: None)

            # Initial display at m=5
            m_val = max(1, cv2.getTrackbarPos('m', win_name))
            ksize = 2 * m_val + 1
            blurred = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
            cv2.imshow(win_name, blurred)

            while True:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                m_val = max(1, cv2.getTrackbarPos('m', win_name))
                ksize = 2 * m_val + 1
                blurred = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
                cv2.imshow(win_name, blurred)
                if (cv2.waitKey(30) & 0xFF) == 27:  # ESC closes
                    break
            cv2.destroyWindow(win_name)

        threading.Thread(target=popup_worker, args=(self.image.copy(),), daemon=True).start()
    
    def show_bilateral_filter(self):
        """Popup: OpenCV trackbar controls bilateral filter radius m (1..5) on currently loaded image"""
        import threading
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        def popup_worker(img):
            win_name = 'Bilateral Filter (m=1..5, d=2m+1, sigma=90)'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 900, 600)

            # Trackbar m in [1,5]
            cv2.createTrackbar('m', win_name, 5, 5, lambda v: None)

            sigma_color = 90
            sigma_space = 90

            # Initial display at m=5 -> d = 11
            m_val = max(1, cv2.getTrackbarPos('m', win_name))
            d = 2 * m_val + 1
            filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
            cv2.imshow(win_name, filtered)

            while True:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                m_val = max(1, cv2.getTrackbarPos('m', win_name))
                d = 2 * m_val + 1
                filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
                cv2.imshow(win_name, filtered)
                if (cv2.waitKey(30) & 0xFF) == 27:  # ESC
                    break
            cv2.destroyWindow(win_name)

        threading.Thread(target=popup_worker, args=(self.image.copy(),), daemon=True).start()
    
    def show_median_filter(self):
        """Popup: OpenCV trackbar controls median filter radius m (1..5) on currently loaded image"""
        import threading
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        def popup_worker(img):
            win_name = 'Median Filter (m=1..5, kernel=(2m+1))'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 900, 600)

            # Trackbar m in [1,5]
            cv2.createTrackbar('m', win_name, 5, 5, lambda v: None)

            # Initial show at m=5 -> kernel 11
            m_val = max(1, cv2.getTrackbarPos('m', win_name))
            k = 2 * m_val + 1
            filtered = cv2.medianBlur(img, k)
            cv2.imshow(win_name, filtered)

            while True:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                m_val = max(1, cv2.getTrackbarPos('m', win_name))
                k = 2 * m_val + 1
                filtered = cv2.medianBlur(img, k)
                cv2.imshow(win_name, filtered)
                if (cv2.waitKey(30) & 0xFF) == 27:  # ESC closes
                    break
            cv2.destroyWindow(win_name)

        threading.Thread(target=popup_worker, args=(self.image.copy(),), daemon=True).start()
    
    # Edge Detection Functions
    def show_sobel_x(self):
        """Popup: Sobel X edge detection (horizontal edges) on loaded image; only show result (no grayscale)."""
        import threading
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        def popup_worker(img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_x_abs = np.absolute(sobel_x)
            sobel_x_8u = np.uint8(sobel_x_abs)

            win_name = 'Sobel X (Horizontal Edges)'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 900, 600)
            cv2.imshow(win_name, sobel_x_8u)
            while True:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if (cv2.waitKey(30) & 0xFF) == 27:
                    break
            cv2.destroyWindow(win_name)

        threading.Thread(target=popup_worker, args=(self.image.copy(),), daemon=True).start()
    
    def show_sobel_y(self):
        """Popup: Use currently loaded image -> gray -> Gaussian blur (3x3, 0,0) -> manual Sobel X (3x3) for horizontal edges. No cv2.Sobel/filter2D."""
        import threading
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        def popup_worker(img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0, 0)
            src = blur.astype(np.int32)

            # Sobel Y kernel detects horizontal edges (changes along Y)
            ky = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.int32)

            padded = np.pad(src, ((1, 1), (1, 1)), mode='edge')
            h, w = src.shape
            out = np.zeros((h, w), dtype=np.int32)
            for y in range(h):
                for x in range(w):
                    roi = padded[y:y+3, x:x+3]
                    out[y, x] = int((roi * ky).sum())

            out_abs = np.clip(np.abs(out), 0, 255).astype(np.uint8)

            win_name = 'Sobel Y (Manual, Horizontal Edges)'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 900, 600)
            cv2.imshow(win_name, out_abs)
            while True:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if (cv2.waitKey(30) & 0xFF) == 27:
                    break
            cv2.destroyWindow(win_name)

        threading.Thread(target=popup_worker, args=(self.image.copy(),), daemon=True).start()
    
    def show_combination_threshold(self):
        """Display three separate windows: Combination, Threshold=128, Threshold=28"""
        import threading
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        def display_single_window(window_name, img, x_pos, y_pos):
            """Display a single image in its own window with independent control"""
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 600, 450)
            cv2.moveWindow(window_name, x_pos, y_pos)
            cv2.imshow(window_name, img)
            
            # Keep window open until closed or ESC is pressed
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key closes this window only
                    break
                # Check if window was closed by user
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            # Close only this specific window
            cv2.destroyWindow(window_name)

        def process_and_display(img_bgr):
            # Convert to grayscale
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Apply Sobel X and Y
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate combination: √(Sobel_X² + Sobel_Y²)
            combination = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize combination to 0-255 range (standard approach)
            combination_normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX)
            combination_normalized = np.uint8(combination_normalized)
            
            # Apply different thresholds to the normalized combination
            _, result_128 = cv2.threshold(combination_normalized, 128, 255, cv2.THRESH_BINARY)
            _, result_28 = cv2.threshold(combination_normalized, 28, 255, cv2.THRESH_BINARY)
            
            # Define window positions to avoid overlapping
            start_x = 100
            start_y = 100
            window_width = 600
            window_height = 450
            spacing = 50
            
            # Create three independent threads for three windows
            thread1 = threading.Thread(
                target=display_single_window,
                args=("Combination of Sobel x and Sobel y", combination_normalized, start_x, start_y),
                daemon=True
            )
            
            thread2 = threading.Thread(
                target=display_single_window,
                args=("O/P (1): Threshold=128", result_128, start_x + window_width + spacing, start_y),
                daemon=True
            )
            
            thread3 = threading.Thread(
                target=display_single_window,
                args=("O/P (2): Threshold=28", result_28, start_x + 2 * (window_width + spacing), start_y),
                daemon=True
            )
            
            # Start all threads
            thread1.start()
            thread2.start()
            thread3.start()

        threading.Thread(target=process_and_display, args=(self.image.copy(),), daemon=True).start()
    
    def show_gradient_angle(self):
        """Display Gradient Angle with specific angle ranges and bitwise operations"""
        import threading
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        def display_single_window(window_name, img, x_pos, y_pos):
            """Display a single image in its own window with independent control"""
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 600, 450)
            cv2.moveWindow(window_name, x_pos, y_pos)
            cv2.imshow(window_name, img)
            
            # Keep window open until closed or ESC is pressed
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key closes this window only
                    break
                # Check if window was closed by user
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            # Close only this specific window
            cv2.destroyWindow(window_name)

        def process_and_display(img_bgr):
            # Convert to grayscale
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Apply Sobel X and Y
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient angle: θ = arctan(Sobel_Y / Sobel_X)
            gradient_angle = np.arctan2(sobel_y, sobel_x)
            gradient_angle_degrees = np.degrees(gradient_angle)
            
            # Convert to 0-360 range for easier angle range checking
            gradient_angle_360 = (gradient_angle_degrees + 360) % 360
            
            # Create mask for range 170°~190°
            mask_170_190 = ((gradient_angle_360 >= 170) & (gradient_angle_360 <= 190)).astype(np.uint8) * 255
            
            # Create mask for range 260°~280°
            mask_260_280 = ((gradient_angle_360 >= 260) & (gradient_angle_360 <= 280)).astype(np.uint8) * 255
            
            # Generate results using cv2.bitwise_and
            result_170_190 = cv2.bitwise_and(gray, mask_170_190)
            result_260_280 = cv2.bitwise_and(gray, mask_260_280)
            
            # Define window positions to avoid overlapping
            start_x = 100
            start_y = 100
            window_width = 600
            window_height = 450
            spacing = 50
            
            # Create two independent threads for two windows
            thread1 = threading.Thread(
                target=display_single_window,
                args=("Gradient Angle (170~190)", result_170_190, start_x, start_y),
                daemon=True
            )
            
            thread2 = threading.Thread(
                target=display_single_window,
                args=("Gradient Angle (260~280)", result_260_280, start_x + window_width + spacing, start_y),
                daemon=True
            )
            
            # Start both threads
            thread1.start()
            thread2.start()

        threading.Thread(target=process_and_display, args=(self.image.copy(),), daemon=True).start()
    
    def apply_transform(self):
        """Apply geometric transformation to burger.png with specific parameters"""
        import threading
        try:
            # Load burger.png
            burger_path = 'Dataset_OpenCvDl_Hw1/Q4_image/burger.png'
            img = cv2.imread(burger_path)
            
            if img is None:
                messagebox.showerror("Error", f"Could not load {burger_path}!")
                return
            
            # Get input values (default values should be set)
            rotation = float(self.rotation_var.get()) if self.rotation_var.get() else 30.0
            scaling = float(self.scaling_var.get()) if self.scaling_var.get() else 0.9
            tx = float(self.tx_var.get()) if self.tx_var.get() else 535.0
            ty = float(self.ty_var.get()) if self.ty_var.get() else 335.0
            
            def display_transformed_image():
                # Get image dimensions (1920x1080)
                h, w = img.shape[:2]
                
                # Original center point of burger: C(240, 200)
                # Target center point: C'(775, 535)
                
                # Create transformation matrix
                # Step 1: Create rotation and scaling matrix around original center (240, 200)
                center = (240, 200)  # Original burger center
                M = cv2.getRotationMatrix2D(center, rotation, scaling)
                
                # Step 2: Calculate translation to move center from (240,200) to (775,535)
                M[0, 2] += 535  # Move X by 535 pixels
                M[1, 2] += 335  # Move Y by 335 pixels
                
                # Apply transformation using cv2.warpAffine
                result = cv2.warpAffine(img, M, (w, h))
                
                # Create OpenCV window
                win_name = 'Transformed Burger'
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 1200, 675)  # Maintain aspect ratio (16:9)
                cv2.imshow(win_name, result)
                
                # Wait for window to be closed or ESC key
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:  # ESC key closes window
                        break
                    # Check if window was closed by user
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                
                # Close window
                cv2.destroyWindow(win_name)
            
            # Run in separate thread to avoid blocking GUI
            threading.Thread(target=display_transformed_image, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for all parameters!")
        except Exception as e:
            messagebox.showerror("Error", f"Transformation failed: {str(e)}")

def main():
    try:
        root = tk.Tk()
        app = ImageProcessorGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("程式被中斷")
    except Exception as e:
        print(f"程式錯誤: {e}")
    finally:
        # 確保清理所有資源
        try:
            cv2.destroyAllWindows()
            plt.close('all')
        except:
            pass
        print("程式已結束")

if __name__ == "__main__":
    main()
