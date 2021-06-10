# -*- coding: 1251 -*-

import sys
sys.path.append('/home/askogan_1')

import glob
import numpy as np
import sys
from time import time
from typing import List, Optional

from bert.embeddings.get_embeddings import init_model
from bert.embeddings.get_embeddings import get_embeddings as batch_to_bert_embeddings


MAX_SENTENCE_LEN = 256  # maximum length of original sentence as a batch for BERT (before tokenization)


def text_to_bert_embeddings(text: str, model: Optional[object] = None, tokenizer: Optional[object] = None, verbose: bool = False) -> List[np.array]:
    """
    Возвращает эмбеддинги BERT для текста
    :param text: преобработанный текст в формате одно предложение на строку (см. `text_preprocess.py`)
    :return: список эмбеддингов размера 128 (по токенам)
    """
    if model is None:
        tokenizer, config, model = init_model('/home/askogan_1/bert/models/uncased_L-12_H-128_A-2')

    if verbose:
        print('text_to_bert_embeddings: loaded model!')
        start_time = time()
        text_size = len(text.split('\n'))
        text_percent = len(text.split('\n')) // 100

    batched_embeddings = []
    for i, sentence in enumerate(text.split('\n')):
        if verbose and i > 0 and i % text_percent == 0:
            print('text_to_bert_embeddings: {}% complete in {:.2f} seconds;  expected {:.2f}'.format(
                i // text_percent, time() - start_time, (time() - start_time) * text_size / i
            ), flush=True)

        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence.split()) > MAX_SENTENCE_LEN:  # разбиваем на подпредложения, если оно слишком большое
            print(f'bert embeddings getter warning: got a sentence with {len(sentence.split())} words; splitting', file=sys.stderr)
            words = sentence.split()
            batches = []
            for i in range(0, len(words), MAX_SENTENCE_LEN):
                batch = ' '.join(words[i:i + MAX_SENTENCE_LEN])
                batches.append(batch)
        else:
            batches = [sentence]

        for batch in batches:  # отправляем [под-]предложения прогоняться через BERT
            batched_embeddings.append(batch_to_bert_embeddings(batch, tokenizer, model))

    return np.vstack(batched_embeddings)


def midtest():
    tokenizer, _, model = init_model('/home/askogan_1/bert/models/bert-base-uncased')
    with open('sample.txt', 'r') as f:
        text = f.read()
    embeddings = text_to_bert_embeddings(text, model, tokenizer)
    print(embeddings.shape)
    for embedding in embeddings:
        print(embedding)


def embed_train_test():
    for folder in ['train', 'val', 'test']:
        with open(f'{folder}/sentences.txt', 'r') as f:
            embeddings = text_to_bert_embeddings(f.read(), verbose=True)
        np.save(f'{folder}/embeddings_128.npy', embeddings)


if __name__ == '__main__':
    embed_train_test()
