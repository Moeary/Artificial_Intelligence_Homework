# predict_emotion.py

import pandas as pd
import jieba
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 加载模型
model_path = "text_emotion_cnn_model.h5"  # 替换为你的模型文件路径
loaded_model = load_model(model_path)

# 加载Tokenizer (需要与训练时使用的Tokenizer相同)
# 假设你已经将Tokenizer保存到文件tokenizer.pickle
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 定义情感标签映射 (需要与训练时使用的映射相同)
label_map = {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}

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
    emotion_confidence = dict(zip(label_map.keys(), confidence)) 
    return emotion_confidence

# 获取用户输入
text = input("请输入要预测的文本: ")

# 进行预测
emotion_confidence = predict_emotion(text)

# 打印结果
print(emotion_confidence) 