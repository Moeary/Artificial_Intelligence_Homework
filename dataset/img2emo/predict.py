import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import gradio as gr
from PIL import Image

# 加载训练好的模型
model = load_model('fer.keras')

# 定义情感分类标签（假设原先有7种标签，数字命名）
CLASS_LABELS = ['1', '2', '3', '4', '5', '6', '7']

# 定义预测函数
def predict_emotion(img):
    img = img.resize((100, 100))  # 调整图像大小
    img_array = np.array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    return predicted_class

# 创建 Gradio 接口
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.inputs.Image(shape=(100, 100)),
    outputs="text",
    title="Emotion Recognition",
    description="Upload an image and the model will predict the emotion."
)

# 启动 Gradio 接口
iface.launch()