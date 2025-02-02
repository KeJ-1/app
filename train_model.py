import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from torch.utils.data import DataLoader

# モデルを事前学習済みResNet18に設定
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 最後の層を2クラスに変更

# データ変換処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像を224x224にリサイズ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 訓練データセットの準備
train_dataset = datasets.ImageFolder(r'C:\Users\keiji\Documents\Fri4\train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# モデル訓練
for epoch in range(5):  # 5エポック訓練
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {running_loss/len(train_loader)}")

# 訓練後にモデルを保存
torch.save(model.state_dict(), 'model.pth')  # model.pthとして保存
