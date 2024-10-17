import json
import csv

# 读取JSON文件
with open('usual_test_labeled.txt', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 写入CSV文件
with open('usual_test_labeled.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'content', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for entry in data:
        writer.writerow(entry)