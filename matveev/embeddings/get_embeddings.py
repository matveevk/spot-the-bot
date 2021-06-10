# -*- coding: 1251 -*-

import argparse
import numpy as np
import re
from scipy import spatial
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (
    BertModel, BertConfig, BertTokenizer, TrainingArguments, 
    Trainer, LineByLineTextDataset, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, DataCollator, BertForMaskedLM, BertForPreTraining
)
from typing import List


def init_model(path='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(path)
    config = BertConfig.from_pretrained(path, output_hidden_states=True)
    model = BertModel.from_pretrained(path, config=config)
    return tokenizer, config, model


def get_ids_tokens(sentence, tokenizer):
    tokens = tokenizer.tokenize('[CLS] ' + sentence + ' [SEP]')
    ids = tokenizer.encode(sentence)
    return ids, tokens


def get_embeddings_from_last(sentence, tokenizer, model):
    """Вытягивает векторные представления для слов в тексте как крайние hidden states"""
    encoded_input = tokenizer(sentence, return_tensors='pt')
    word_embeddings = model(**encoded_input)[2][-1][0].detach().numpy()[1:-1]
    return word_embeddings


def get_embeddings(sentence, tokenizer, model):
    """Вытягивает векторные представления для слов в тексте на основе 4 крайних hidden states"""
    encoded_input = tokenizer(sentence, return_tensors='pt')
    four_last_hidden = model(**encoded_input)[2][-4:]
    four_last_hidden = np.asarray([hidden[0].detach().numpy() for hidden in four_last_hidden])
    word_embeddings = four_last_hidden[:, 1:-1].sum(axis=0)
    return word_embeddings


class Sentence:
    """Холдер под предложения с эмбеддингами"""
    def __init__(self, raw: str, tokens: List[str], embeddings: List[np.array]):
        self.raw = raw
        self.tokens = tokens
        self.embeddings = embeddings


def embed_text(inputfile: str, debug: bool = False) -> List[np.array]:
    """Возвращает эмбеддинги для текста в inputfile"""
    with open(inputfile, 'r') as f:
        raw_sentences = re.split('([.;!?] *)', f.read())

    tokenizer, config, model = init_model('/home/askogan_1/bert/models/uncased_L-12_H-128_A-2')

    embeddings = []
    if debug:
        sentences = []

    for raw_sentence in raw_sentences:
        sentence_embeddings = get_embeddings(raw_sentence, tokenizer, model)
        embeddings.extend(sentence_embeddings)
        if debug:
            sentences.append(Sentence(raw_sentence, tokenizer.tokenize(raw_sentence), sentence_embeddings))

    if debug:
        for sentence in sentences:
            print(sentence.raw)
            for token, embedding in zip(sentence.tokens, sentence.embeddings):
                print(token, embedding.shape, embedding.mean(), embedding[:5])
            print()

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    embed_text(args.filepath, debug=args.debug)
