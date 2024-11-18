import torch
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import torch.nn as nn
import cv2
import numpy as np

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

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.resnet(x)

class GoogLeNetModel(nn.Module):
    def __init__(self):
        super(GoogLeNetModel, self).__init__()
        self.googlenet = models.googlenet(pretrained=True)
        num_ftrs = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.googlenet(x)

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# 预测函数
def predict(image, model):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    emotions = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return {emotions[i]: probabilities[0][i].item() for i in range(len(emotions))}

# 处理视频
def process_video(input_video_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "FastCNN":
        model = FastCNN().to(device)
        model.load_state_dict(torch.load('model/img2emo/fast-cnn_model.pth', map_location=device, weights_only=True))
    elif model_name == "ResNet":
        model = ResNetModel().to(device)
        model.load_state_dict(torch.load('model/img2emo/resnet_model.pth', map_location=device, weights_only=True))
    elif model_name == "GoogLeNet":
        model = GoogLeNetModel().to(device)
        model.load_state_dict(torch.load('model/img2emo/googlenet_model.pth', map_location=device, weights_only=True))
    
    model.eval()

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = input_video_path.replace('.mp4', '_output.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        probabilities = predict(frame, model)
        y0, dy = 30, 30
        for i, (emotion, prob) in enumerate(probabilities.items()):
            text = f'{emotion}: {prob * 100:.2f}%'
            y = y0 + i * dy
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    return output_video_path

# Gradio界面
video_input = gr.Video(label="Input Video")
model_input = gr.Radio(choices=["ResNet", "GoogLeNet", "FastCNN"], label="Select Model")
video_output = gr.Video(label="Output Video")

def process_video_gradio(input_video_path, model_name):
    output_video_path = process_video(input_video_path, model_name)
    return output_video_path

gr.Interface(fn=process_video_gradio, inputs=[video_input, model_input], outputs=video_output, title="Video Emotion Prediction", description="Upload a video and select a model to predict emotions in the video.").launch()