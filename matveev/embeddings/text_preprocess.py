# -*- coding: 1251 -*-

import argparse
import en_core_web_lg
# import nltk.data
import re
import spacy
import sys

from typing import List, Optional


class SentenceTokenizer:
    """Source: https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences"""

    alphabets= "([A-Za-z])"
    digits = "([0-9])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    def split_into_sentences(self, text: str) -> List[str]:
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(self.prefixes,"\\1<prd>",text)
        text = re.sub(self.websites,"<prd>\\1",text)
        if "Ph.D" in text:
            text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + self.alphabets + "[.] "," \\1<prd> ", text)
        text = re.sub(self.acronyms + " " + self.starters,"\\1<stop> \\2", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]","\\1<prd>\\2<prd>", text)
        text = re.sub(self.digits + "[.]" + self.digits,"\\1<prd>\\2", text)
        text = re.sub(" " + self.suffixes + "[.] " + self.starters, " \\1<stop> \\2", text)
        text = re.sub(" " + self.suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + self.alphabets + "[.]"," \\1<prd>", text)
        if "”" in text:
            text = text.replace(".”","”.")
        if "\"" in text:
            text = text.replace(".\"","\".")
        if "!" in text:
            text = text.replace("!\"","\"!")
        if "?" in text:
            text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences


spacy_pos = {'PRON': 'HE', 'NUM': 'NUMBER'}
spacy_ent2word = {
    'CARDINAL': 'NUMBER',
    'DATE': 'DATE',
    'EVENT': 'EVENT',
    'FAC': 'FACILITY',
    'GPE': 'COUNTRY',
    'LANGUAGE': 'LANGUAGE',
    'LAW': 'LAW',
    'LOC': 'LOCATION',
    'MONEY': 'MONEY',
    'NORP': 'NATION',
    'ORDINAL': 'FIRST',
    'ORG': 'COMPANY',
    'PERCENT': 'PERCENT',
    'PERSON': 'PERSON',
    'PRODUCT': 'PRODUCT',
    'QUANTITY': 'QUANTITY',
    'TIME': 'TIME',
    'WORK_OF_ART': 'ART',
}


def decontracted(phrase: str, sentence_splitter: Optional[object] = None) -> str:
    # if sentence_splitter is None:
    #     # source: https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    #     sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')

    # specific
    phrase = re.sub(r"won['’‘`]t", "will not", phrase)
    phrase = re.sub(r"can['’‘`]t", "can not", phrase)
    phrase = re.sub(r"ain['’‘`]t", "am not", phrase)

    # general
    phrase = re.sub(r"n['’‘`]t", " not", phrase)
    phrase = re.sub(r"['’‘`]re", " are", phrase)
    phrase = re.sub(r"['’‘`]s", " is", phrase)  # "my mother's eyes"?
    phrase = re.sub(r"['’‘`]d", " would", phrase)
    phrase = re.sub(r"['’‘`]ll", " will", phrase)
    phrase = re.sub(r"['’‘`]t", " not", phrase)  # ?
    phrase = re.sub(r"['’‘`]ve", " have", phrase)
    phrase = re.sub(r"['’‘`]m", " am", phrase)

    phrase = re.sub(r'[^\w.?!;]', ' ', phrase)	
    phrase = re.sub(' +', ' ', phrase)

    # return re.sub(r'(?<!\w\.\w.)(?<=\.|\?|\!\;)\s', '\n', phrase)  # source: https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
    # sentences = sentence_splitter.tokenize(phrase.strip())
    sentences = re.split('[.;!?]\s*', phrase)
    return '\n'.join([sentence.strip() for sentence in sentences if sentence.strip()])


def preprocess_english_text(text: str, nlp: Optional[object] = None) -> str:
    """
    Препроцессинг текста на английском: заменяет в тексте Named Entity и pronouns (см. словари)
    Текст склеивается несколько строк через \n (одна строка на предложение)
    """
    if nlp is None:
        nlp = en_core_web_lg.load(disable=['parser'])
        nlp.max_length = 5000000

    decontracted_text = decontracted(text)
    decontracted_sentences = decontracted_text.split('\n')

    preprocessed_sentences = []
    for sentence in decontracted_sentences[:2000]:
        # print(f'sentence "{sentence}"')
        nlp_sentence = nlp(sentence)

        preprocessed_words = []
        for token in nlp_sentence:
            # print('\t', token, token.text, token.pos_, token.lemma_, token.ent_type_)
            if token.ent_type_ != '':
                preprocessed_words.append(token.text)
                continue
            if token.pos_ == 'PUNCT':
                continue

            preprocessed_word = token.text
            if token.pos_ in spacy_pos:
                preprocessed_word = spacy_pos[token.pos_]
            elif token.lemma_[1:-1] in spacy_pos:  # например, "my" -> "-PRON-"
                preprocessed_word = spacy_pos[token.lemma_[1:-1]]
            else:
                if not token.lemma_.islower() and token.pos_ != 'SPACE':
                    # print('english preprocessing warning: token.lemma_ has upper case --- ', token, token.lemma_, token.pos_, file=sys.stderr)
                    pass
                preprocessed_word = token.lemma_.lower()
            preprocessed_words.append(preprocessed_word)

        preprocessed_sentence = ' '.join(preprocessed_words)

        sorted_ents = sorted(nlp_sentence.ents, key=len, reverse=True)
        for ent in sorted_ents:
            # print('entity found:', ent.text, ent.label_)
            preprocessed_sentence = preprocessed_sentence.replace(' ' + ent.text + ' ', ' ' + spacy_ent2word[ent.label_] + ' ')

        preprocessed_sentence = re.sub(r'\s\s+', ' ', preprocessed_sentence)
        preprocessed_sentences.append(preprocessed_sentence.strip())

    return '\n'.join(preprocessed_sentences)


def preprocess_english_file(input_path: str, output_path: str) -> None:
    with open(input_path, 'r') as input_file:
        text = input_file.read()
    result_text = preprocess_english_text(text)
    with open(output_path, 'w') as output_file:
        output_file.write(result_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile')
    parser.add_argument('-o', '--outputfile')
    args = parser.parse_args()
    preprocess_english_file(args.inputfile, args.outputfile)
