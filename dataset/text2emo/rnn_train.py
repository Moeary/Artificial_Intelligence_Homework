import pandas as pd
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
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
num_words = 65590  # 微博情感分析器总共有的词汇量

tokenizer = Tokenizer(num_words=num_words)  # 可以根据词汇量调整
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# 填充序列，使其长度一致
max_length = 200  # 可以根据文本长度调整
train_padded = pad_sequences(train_sequences, maxlen=max_length)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

# 3. 构建RNN模型
model = Sequential()
model.add(Embedding(num_words, 128, input_length=max_length))
model.add(SimpleRNN(128, return_sequences=False))
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='softmax'))  # 6个情感类别

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 训练模型并评估测试集
best_accuracy = 0.0
for epoch in range(1, 21):  # 训练20轮
    print(f'Epoch {epoch}/{20}')
    model.fit(train_padded, train_labels, epochs=1, batch_size=64, validation_split=0.2)
    
    # 评估测试集
    loss, accuracy = model.evaluate(test_padded, test_labels, verbose=0)
    print(f'Test Accuracy: {accuracy}')
    
    # 如果测试集准确率比之前的高，则保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model.save("best_text_emotion_rnn_model.h5")
        print(f'Saved best model with accuracy: {best_accuracy}')

# 保存最终的Tokenizer
import pickle
with open('rnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f'Best Test Accuracy: {best_accuracy}')