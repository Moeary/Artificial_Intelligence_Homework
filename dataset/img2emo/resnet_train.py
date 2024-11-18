import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# 加载训练和测试数据集
train_dataset = datasets.ImageFolder(root='RAF-DB DATASET/train', transform=transform)
test_dataset = datasets.ImageFolder(root='RAF-DB DATASET/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的ResNet模型并进行微调
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 6)  # 假设有6个类别

    def forward(self, x):
        return self.resnet(x)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Train Epoch: {epoch}\tAverage Loss: {train_loss:.6f}\tAccuracy: {train_acc:.2f}%')

def test(model, device, test_loader, criterion, best_acc, save_path):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n')

    # Save the model if test accuracy is the best
    if test_acc > best_acc:
        print(f'Saving model with accuracy {test_acc:.2f}%')
        torch.save(model.state_dict(), save_path)
        best_acc = test_acc

    return best_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNetModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
save_path = 'model/resnet_model.pth'
isExists = os.path.exists('model')
if not isExists:
    os.makedirs('model')
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    best_acc = test(model, device, test_loader, criterion, best_acc, save_path)