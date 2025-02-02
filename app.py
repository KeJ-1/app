import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from torch import nn
from torchvision import models

# モデルのロード
model = models.resnet18(weights='IMAGENET1K_V1')  # 事前学習モデルを使う
model.fc = nn.Linear(model.fc.in_features, 2)  # 出力を猫と犬の2クラスに設定
model.load_state_dict(torch.load('model.pth'))  # 学習済みモデルのロード
model.eval()  # 推論モードに設定

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# tkinter アプリケーションの作成
def upload_image():
    file_path = filedialog.askopenfilename(title="画像を選択", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
    
    if file_path:
        # 画像の表示
        img = Image.open(file_path)
        img.thumbnail((250, 250))  # サイズ調整
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        
        # 画像の分類
        predict_image(file_path)

def predict_image(image_path):
    # 画像の読み込みと前処理
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)  # バッチサイズ1に変換
    
    # 推論
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    # 結果表示
    if predicted == 0:
        result = "猫"
    else:
        result = "犬"
    
    messagebox.showinfo("予測結果", f"この画像は {result} です。")

# GUIの作成
root = tk.Tk()
root.title("猫か犬か判別アプリ")

# ウィンドウサイズを設定
root.geometry("400x200")

# 画像表示エリア
panel = tk.Label(root)
panel.pack(pady=10)

# アップロードボタン
upload_button = tk.Button(root, text="画像をアップロード", command=upload_image)
upload_button.pack(pady=10)

# メインループの開始
root.mainloop()
