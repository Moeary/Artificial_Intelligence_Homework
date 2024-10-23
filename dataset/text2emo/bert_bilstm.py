import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional

# 1. 数据加载和预处理
train_df = pd.read_csv("usual_train.csv", encoding='utf-8')
test_df = pd.read_csv("usual_test_labeled.csv", encoding='utf-8')

# 数据清洗：将content列转换为字符串
train_df['content'] = train_df['content'].astype(str)
test_df['content'] = test_df['content'].astype(str)

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
# 使用BERT的Tokenizer将文本转换为数字序列
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])

    return np.array(input_ids), np.array(attention_masks)

max_length = 200  # 可以根据文本长度调整
train_input_ids, train_attention_masks = encode_texts(train_texts, tokenizer, max_length)
test_input_ids, test_attention_masks = encode_texts(test_texts, tokenizer, max_length)

# 3. 构建BERT-BiLSTM模型
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# 冻结BERT模型的所有层
for layer in bert_model.layers:
    layer.trainable = False

input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
attention_masks = Input(shape=(max_length,), dtype=tf.int32, name='attention_masks')

bert_output = bert_model(input_ids, attention_mask=attention_masks)[0]
lstm_output = Bidirectional(LSTM(128, return_sequences=False))(bert_output)
dropout = Dropout(0.3)(lstm_output)
output = Dense(6, activation='softmax')(dropout)

model = Model(inputs=[input_ids, attention_masks], outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), metrics=['accuracy'])

# 4. 训练模型
model.fit([train_input_ids, train_attention_masks], train_labels, epochs=4, batch_size=16, validation_split=0.2)

# 保存模型
model.save("text_emotion_bert_bilstm_model.h5")

# 保存Tokenizer
import pickle
with open('bert_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 5. 评估模型
loss, accuracy = model.evaluate([test_input_ids, test_attention_masks], test_labels, verbose=0)
print('Test Accuracy: {}'.format(accuracy))