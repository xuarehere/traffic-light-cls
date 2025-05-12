'''
Author: xuarehere xuarehere@foxmail.com
Date: 2025-04-09 15:27:33
LastEditTime: 2025-05-12 17:30:04
LastEditors: xuarehere xuarehere@foxmail.com
Description: 
FilePath: /traffic-light-cls/train.py

'''
# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet import get_resnet
from models.small_resnet import SmallResNet, load_pretrained_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型
model = get_resnet(num_classes=3).to(device)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# 保存模型
torch.save(model.state_dict(), 'resnet_traffic_light.pth')
