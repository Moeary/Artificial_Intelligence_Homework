import jieba
import pandas as pd

# 读取数据
data = pd.read_csv("./dataset/text2emo/usual_train.csv", encoding='utf-8')

# 使用jieba分词
hash_count = {}
for text in data['content']:
    # Convert to string to avoid AttributeError
    text = str(text)
    words = jieba.cut(text)
    for word in words:
        if word in hash_count:
            hash_count[word] += 1
        else:
            hash_count[word] = 1

print("Total words:", len(hash_count))