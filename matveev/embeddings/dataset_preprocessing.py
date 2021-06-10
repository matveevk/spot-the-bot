# -*- coding: 1251 -*-


import sys
sys.path.append('/home/askogan_1')

import argparse
import en_core_web_lg
import re
import os
import spacy
import sys
from glob import glob
from time import time

from bert.embeddings.text_preprocess import preprocess_english_text


def preprocess_english_dataset(inputdir: str, outputdir: str, nlp: object) -> None:
    start = time()
    filenames = glob(inputdir + '/*/*.txt')
    for i, filename in enumerate(filenames):
        if i:
            print(f'processed {i}\ttime passed {time() - start},\texpected{(time() - start) * len(filenames) / i}', flush=True)
        with open(filename, 'r') as f:
            preprocessed_text = preprocess_english_text(f.read(), nlp)
        target_filename = filename.replace('EnLit', 'EnLitPreprocessed')
        os.makedirs(os.path.dirname(target_filename), exist_ok=True)
        with open(target_filename, 'w') as f:
            f.write(preprocessed_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir')
    parser.add_argument('-o', '--outputdir')
    args = parser.parse_args()

    nlp = en_core_web_lg.load(disable=['parser'])
    nlp.max_length = 5000000

    preprocess_english_dataset(args.inputdir, args.outputdir, nlp)
