import pandas as pd
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 数据加载和预处理
train_df = pd.read_csv("usual_train.csv", encoding='utf-8')
test_df = pd.read_csv("usual_test_labeled.csv", encoding='utf-8')

# 数据清洗：将content列转换为字符串
train_df['content'] = train_df['content'].astype(str)
test_df['content'] = test_df['content'].astype(str)

# 使用jieba进行分词
train_df['content'] = train_df['content'].apply(lambda x: " ".join(jieba.cut(x)))
test_df['content'] = test_df['content'].apply(lambda x: " ".join(jieba.cut(x)))

# 将标签转换为数字
label_map = {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}
train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# 划分训练集和验证集
train_texts = train_df['content'].values
train_labels = train_df['label'].values
test_texts = test_df['content'].values
test_labels = test_df['label'].values

# 2. 文本向量化
# 使用Tokenizer将文本转换为数字序列
nums_words = 65590  # 微博情感分析器总共有的词汇量

tokenizer = Tokenizer(nums_words)  # 可以根据词汇量调整
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# 填充序列，使其长度一致
max_length = 200  # 可以根据文本长度调整
train_padded = pad_sequences(train_sequences, maxlen=max_length)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

# 3. 构建CNN模型
model = Sequential()
model.add(Embedding(nums_words, 128, input_length=max_length))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='softmax'))  # 6个情感类别

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 训练模型 4轮之后开始出现过拟合现象 就取epoch=4
model.fit(train_padded, train_labels, epochs=4, batch_size=64, validation_split=0.2) 

# 保存模型
model.save("text_emotion_cnn_model.h5") 

# 保存Tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 5. 评估模型
loss, accuracy = model.evaluate(test_padded, test_labels, verbose=0)
print('Test Accuracy: {}'.format(accuracy))
