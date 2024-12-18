import whisper
import torch
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import jieba
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from transformers import TFBertModel, BertTokenizer
import subprocess

# 加载Whisper模型
whisper_model = whisper.load_model("base")

# 定义情感标签映射 (需要与训练时使用的映射相同)
label_map = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'surprise', 4: 'sad', 5: 'fear'}

# 设置最大序列长度 (需要与训练时使用的最大长度相同)
max_length = 200

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

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

# 预测图像情绪
def predict_image_emotion(image, model):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    return {i: probabilities[0][i].item() for i in range(len(probabilities[0]))}

# 预测文本情绪
def predict_text_emotion(text, text_model_name):
    # 加载情绪识别模型
    model_path = f"model/text2emo/{text_model_name}_model.h5"
    if text_model_name == "bert-bilstm":
        loaded_model = load_model(model_path, custom_objects={"TFBertModel": TFBertModel})
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # 分词和向量化
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length', return_tensors='tf')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        text_padded = [input_ids, attention_mask]
    else:
        loaded_model = load_model(model_path)
        # 加载Tokenizer (需要与训练时使用的Tokenizer相同)
        tokenizer_path = f"model/text2emo/{text_model_name}_tokenizer.pickle"
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        # 分词
        text_seq = " ".join(jieba.cut(text))
        # 向量化
        text_seq = tokenizer.texts_to_sequences([text_seq])
        text_padded = pad_sequences(text_seq, maxlen=max_length)
    
    # 预测情绪
    prediction = loaded_model.predict(text_padded)
    # 获取每个标签的置信度
    confidence = prediction[0]
    # 创建标签和置信度的字典
    return {i: float(confidence[i]) for i in range(len(confidence))}

# 处理视频
def process_video(input_video_path, image_model_name, text_model_name, text_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if image_model_name == "FastCNN":
        model = FastCNN().to(device)
        model.load_state_dict(torch.load('model/img2emo/fast-cnn_model.pth', map_location=device, weights_only=True))
    elif image_model_name == "ResNet":
        model = ResNetModel().to(device)
        model.load_state_dict(torch.load('model/img2emo/resnet_model.pth', map_location=device, weights_only=True))
    elif image_model_name == "GoogLeNet":
        model = GoogLeNetModel().to(device)
        model.load_state_dict(torch.load('model/img2emo/googlenet_model.pth', map_location=device, weights_only=True))
    
    model.eval()

    # 提取音频
    audio_path = input_video_path.replace('.mp4', '.wav')
    subprocess.run(['ffmpeg', '-y', '-i', input_video_path, '-q:a', '0', '-map', 'a', audio_path], check=True)

    # 使用Whisper模型进行语音识别
    result = whisper_model.transcribe(audio_path, language="zh")
    segments = result["segments"]

    # 创建一个字典来存储每个时间段的文本情绪
    text_emotions_dict = {}
    for segment in segments:
        text = segment["text"]
        print(text)
        start_time = segment["start"]
        end_time = segment["end"]
        text_emotions = predict_text_emotion(text, text_model_name)
        for t in np.arange(start_time, end_time, 1.0 / 30.0):  # 假设视频帧率为30fps
            text_emotions_dict[t] = text_emotions

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = input_video_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 图像情绪识别
        image_emotions = predict_image_emotion(frame, model)

        # 获取当前帧的时间戳
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 获取当前时间段的文本情绪
        text_emotions = text_emotions_dict.get(current_time, {i: 0 for i in range(len(label_map))})

        # 加权合并情绪结果
        combined_emotions = {}
        for i in range(len(label_map)):
            combined_emotions[i] = (image_emotions[i] * (1 - text_weight)) + (text_emotions[i] * text_weight)

        y0, dy = 30, 30
        for i, (emotion, prob) in enumerate(combined_emotions.items()):
            emotion_str = label_map[emotion]
            text = f'{emotion_str}: {prob * 100:.2f}%'
            y = y0 + i * dy
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

    # 将音频添加回视频
    output_video_path = input_video_path.replace('.mp4', '_output.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_video_path], check=True)


    return output_video_path

# Gradio界面
video_input = gr.Video(label="Input Video")
image_model_input = gr.Radio(choices=["ResNet", "GoogLeNet", "FastCNN"], label="Select Image Model",value="ResNet")
text_model_input = gr.Radio(choices=["cnn", "rnn", "bert-bilstm"], label="Select Text Model",value="cnn")
text_weight_input = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.333, label="Text Weight")
video_output = gr.Video(label="Output Video")

def process_video_gradio(input_video_path, image_model_name, text_model_name, text_weight):
    output_video_path = process_video(input_video_path, image_model_name, text_model_name,text_weight)
    return output_video_path

gr.Interface(fn=process_video_gradio, inputs=[video_input, image_model_input, text_model_input,text_weight_input], outputs=video_output, title="Video Emotion Prediction", description="Upload a video and select models to predict emotions in the video.").launch()