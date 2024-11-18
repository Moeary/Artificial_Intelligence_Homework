# predict_emotion.py

import pandas as pd
import jieba
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import pickle

# 加载模型
model_path = "model\\text2emo\\text_emotion_cnn_model.h5"  # 替换为你的模型文件路径
loaded_model = load_model(model_path)

# 加载Tokenizer (需要与训练时使用的Tokenizer相同)
# 假设你已经将Tokenizer保存到文件tokenizer.pickle
with open('model/text2emo/cnn_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 定义情感标签映射 (需要与训练时使用的映射相同)
label_map = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'surprise', 4: 'sad', 5: 'fear'}

# 设置最大序列长度 (需要与训练时使用的最大长度相同)
max_length = 200

# 定义预测函数
def predict_emotion(text):
    # 分词
    text_seq = " ".join(jieba.cut(text))
    # 向量化
    text_seq = tokenizer.texts_to_sequences([text_seq])
    text_padded = pad_sequences(text_seq, maxlen=max_length)
    # 预测
    prediction = loaded_model.predict(text_padded)
    # 获取每个标签的置信度
    confidence = prediction[0]
    # 创建标签和置信度的字典
    emotion_confidence = {label_map[i]: float(confidence[i]) for i in range(len(confidence))}
    return emotion_confidence

# Gradio界面
text_input = gr.Textbox(lines=2, placeholder="请输入要预测的文本")
output = gr.Label(num_top_classes=6)

gr.Interface(fn=predict_emotion, inputs=text_input, outputs=output, title="Text Emotion Prediction", description="Please enter your text,it will auto predict emotion probility").launch()