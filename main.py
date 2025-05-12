import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
# 参数配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "traffic_light_data"  # 数据集文件夹路径
output_model_path = "traffic_light_classifier.pt"  # 模型保存路径
batch_size = 32
epochs = 20
learning_rate = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class TrafficLightClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TrafficLightClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = TrafficLightClassifier(num_classes=len(train_dataset.classes))
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train_model():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), output_model_path)

# 测试模型
def test_model():
    model.load_state_dict(torch.load(output_model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

# 预测单张图片的类别
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    class_names = train_dataset.classes
    return class_names[predicted.item()]

if __name__ == "__main__":
    train_model()
    test_model()
    model.load_state_dict(torch.load(output_model_path))
    model.eval()
    # path = r'./test/green'
    # path = r'./test/yellow'
    path = r'./test/red'
    # for root, dirs, files in os.walk(r"./test - 副本"):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"file:{file_path}\t\tPredicted class: {predict_image(file_path, model)}")