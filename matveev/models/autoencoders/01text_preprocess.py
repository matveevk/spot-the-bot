# -*- coding: 1251 -*-

import sys
sys.path.append('/home/askogan_1')

import argparse
import glob
from bert.embeddings.text_preprocess import preprocess_english_text


def preprocess_english_files() -> None:
    input_files = glob.glob('full_raw/*.txt')
    for filepath in input_files:
        with open(filepath, 'r') as input_file:
            text = input_file.read()
        result_text = preprocess_english_text(text)
        filepath = filepath.replace('_raw', '_preprocessed')
        with open(filepath, 'w') as output_file:
            output_file.write(result_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    preprocess_english_files()
