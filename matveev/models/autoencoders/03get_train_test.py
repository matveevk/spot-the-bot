# -*- coding: 1251 -*-


"""
Выбирает рандомно train-test-val выборку для автоенкодера
из всех отобранных и препроцессированных предложений 
в файле `sticked_file.txt`.
Сохраняет в файлах `train_sentences.txt`, `test_sentences.txt`, `val_sentences`.
"""


import numpy as np
from sklearn.model_selection import train_test_split


TRAIN_SZ = 200_000  # in sentences
TEST_SZ = 20_000
VAL_SZ = 20_000


np.random.seed(1984)


# loading all sentences from file into array
sentences = []
with open('sticked_file.txt', 'r') as f:
    for sent in f:
        sent = sent.strip()
        if sent:
            sentences.append(sent)


# choosing indicies for dataset
all_idx = np.random.choice(len(sentences), size=TRAIN_SZ + TEST_SZ + VAL_SZ, replace=False)  # expecting approx. 1 million tokens, already around six gb of memory

train_idx, test_idx = train_test_split(all_idx, train_size=TRAIN_SZ, random_state=2000)
val_idx, test_idx = train_test_split(test_idx, train_size=VAL_SZ, random_state=2036)


# sorting for quicker operations (remove?)
train_idx = np.sort(train_idx)
val_idx = np.sort(val_idx)
test_idx = np.sort(test_idx)


a, b, c = 0, 0, 0
x, y, z = 0, 0, 0


# writing train
train_word_cnt = 0
with open('train/sentences.txt', 'w') as f:
    for i in train_idx:
        train_word_cnt += len(sentences[i].split())
        a = max(a, len(sentences[i].split()))
        x = max(x, len(sentences[i]))
        f.write(sentences[i])
        f.write('\n')

# writing val
val_word_cnt = 0
with open('val/sentences.txt', 'w') as f:
    for i in val_idx:
        val_word_cnt += len(sentences[i].split())
        b = max(b, len(sentences[i].split()))
        y = max(y, len(sentences[i]))
        f.write(sentences[i])
        f.write('\n')

# writing test
test_word_cnt = 0
with open('test/sentences.txt', 'w') as f:
    for i in test_idx:
        test_word_cnt += len(sentences[i].split())
        c = max(c, len(sentences[i].split()))
        z = max(z, len(sentences[i]))
        f.write(sentences[i])
        f.write('\n')

print(f'train word count: {train_word_cnt}')
print(f'val word count: {val_word_cnt}')
print(f'test word count: {test_word_cnt}')
print()
print(f'largest sentence len (word count): {a}/{b}/{c}')
print(f'largest sentence len (char count): {x}/{y}/{z}')
