import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import torch.nn as nn

class FastCNN(nn.Module):
    def __init__(self):
        super(FastCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model = FastCNN().to(device)
model.load_state_dict(torch.load('model/fast-cnn_model.pth', weights_only=True))
model.eval()

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# 预测函数
def predict(image_paths):
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    image_tensors = torch.stack([transform(image) for image in images]).to(device)
    with torch.no_grad():
        outputs = model(image_tensors)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    emotions = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    predicted_emotions = [emotions[p.item()] for p in predicted]
    return images, probabilities, predicted_emotions

# 可视化函数
def visualize_predictions(images, predicted_emotions):
    for image, predicted_emotion in zip(images, predicted_emotions):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = f'Predicted: {predicted_emotion}'
        draw.text((10, 10), text, fill="red", font=font)
        image.show()

# 对测试集目录下的所有图像进行分类并可视化
test_dir = 'RAF-DB DATASET/test'
image_paths = [os.path.join(root, file) for root, _, files in os.walk(test_dir) for file in files if file.endswith(('jpg', 'jpeg', 'png'))]

batch_size = 32
count = 0
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    images, probabilities, predicted_emotions = predict(batch_paths)
    for image_path, predicted_emotion in zip(batch_paths, predicted_emotions):
        #print(f'Image: {image_path}')
        print(f'Predicted Emotion: {predicted_emotion}')
        count += 1
        print(f'Processed {count}/{len(image_paths)} images')
    #visualize_predictions(images, predicted_emotions)