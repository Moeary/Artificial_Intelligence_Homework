import pandas as pd
import jieba
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import pickle
from transformers import TFBertModel, BertTokenizer

# 定义情感标签映射 (需要与训练时使用的映射相同)
label_map = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'surprise', 4: 'sad', 5: 'fear'}

# 设置最大序列长度 (需要与训练时使用的最大长度相同)
max_length = 200

# 定义预测函数
def predict_emotion(text, model_name):
    # 加载模型
    model_path = f"model/text2emo/{model_name}_model.h5"
    if model_name == "bert-bilstm":
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
        tokenizer_path = f"model/text2emo/{model_name}_tokenizer.pickle"
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
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
model_input = gr.Radio(choices=["cnn", "rnn", "bert-bilstm"], label="Select Model")
output = gr.Label(num_top_classes=6)

gr.Interface(fn=predict_emotion, inputs=[text_input, model_input], outputs=output, title="Text Emotion Prediction", description="Please enter your text and select a model to predict emotion probability.").launch()