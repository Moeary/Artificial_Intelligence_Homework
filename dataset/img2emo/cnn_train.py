import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 数据路径
train_dir = "./dataset/train"
test_dir = "./dataset/test"

# 超参数
IMG_HEIGHT = 75
IMG_WIDTH = 75
BATCH_SIZE = 64
EPOCHS = 30
NUM_CLASSES = 6
CLASS_LABELS = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_HEIGHT),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(train_dir), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(train_dir), data_transforms['val']),
    'test': datasets.ImageFolder(os.path.join(test_dir), data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

# 定义模型
def create_resnet34_model(num_classes):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_googlenet_model(num_classes):
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

class FastCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 创建模型
resnet34_model = create_resnet34_model(NUM_CLASSES).to(device)
googlenet_model = create_googlenet_model(NUM_CLASSES).to(device)
fast_cnn_model = FastCNN(NUM_CLASSES).to(device)

models = [resnet34_model, googlenet_model, fast_cnn_model]
model_names = ["ResNet34", "GoogLeNet", "Fast-CNN"]

# 训练和评估模型
def train_model(model, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                # 只在训练阶段计算梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化，只在训练阶段进行
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

def main():
    for model, name in zip(models, model_names):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"Training {name} model...")
        model = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
        torch.save(model.state_dict(), f"{name}_model.pth")
        print(f"{name} model saved.")

    # 评估模型
    def evaluate_model(model, dataloader):
        model.eval()
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

        accuracy = running_corrects.double() / dataset_sizes['test']
        return accuracy

    for model, name in zip(models, model_names):
        accuracy = evaluate_model(model, dataloaders['test'])
        print(f"{name} Test Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()