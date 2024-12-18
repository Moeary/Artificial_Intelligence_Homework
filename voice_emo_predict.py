import whisper
import pandas as pd
import jieba
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import pickle
from transformers import TFBertModel, BertTokenizer

# 加载Whisper模型
whisper_model = whisper.load_model("base")

# 定义情感标签映射 (需要与训练时使用的映射相同)
label_map = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'surprise', 4: 'sad', 5: 'fear'}

# 设置最大序列长度 (需要与训练时使用的最大长度相同)
max_length = 200

# 定义预测函数
def predict_emotion(audio, model_name):
    # 使用Whisper模型进行语音识别
    result = whisper_model.transcribe(audio, language="zh")
    segments = result["segments"]
    
    # 加载情绪识别模型
    model_path = f"model/text2emo/{model_name}_model.h5"
    if model_name == "bert-bilstm":
        loaded_model = load_model(model_path, custom_objects={"TFBertModel": TFBertModel})
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    else:
        loaded_model = load_model(model_path)
        tokenizer_path = f"model/text2emo/{model_name}_tokenizer.pickle"
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    
    # 构建SRT格式的输出
    srt_output = ""
    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # 格式化时间为SRT格式
        start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
        end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"
        
        # 分词和向量化
        if model_name == "bert-bilstm":
            inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length', return_tensors='tf')
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            text_padded = [input_ids, attention_mask]
        else:
            text_seq = " ".join(jieba.cut(text))
            text_seq = tokenizer.texts_to_sequences([text_seq])
            text_padded = pad_sequences(text_seq, maxlen=max_length)
        
        # 预测情绪
        prediction = loaded_model.predict(text_padded)
        confidence = prediction[0]
        emotion_confidence = {label_map[i]: float(confidence[i]) for i in range(len(confidence))}
        
        # 构建SRT条目
        srt_output += f"{i + 1}\n{start_time_str} --> {end_time_str}\n"
        for emotion, prob in emotion_confidence.items():
            srt_output += f"{emotion}: {prob * 100:.2f}%\n"
        srt_output += "\n"
    
    return srt_output

# Gradio界面
audio_input = gr.Audio(type="filepath", label="Speak or Upload Audio")
model_input = gr.Radio(choices=["cnn", "rnn", "bert-bilstm"], label="Select Model")
output = gr.Textbox(label="Transcribed Text (SRT Format)")

gr.Interface(fn=predict_emotion, inputs=[audio_input, model_input], outputs=output, title="Voice Emotion Prediction", description="Speak into the microphone or upload an audio file to transcribe it to text using Whisper and predict emotion using the selected model. The output will be in SRT format.").launch()