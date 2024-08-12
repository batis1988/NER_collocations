# a small guide to prepare raw specified data

import json
import pickle
import re
import pandas as pd
from ast import literal_eval

# read and clean the corpus

temp_file = open("./data/ner.json")
ner_dct = json.load(temp_file)
# read corpus
raw_corpus = pd.read_csv("./data/train.csv", usecols=[1, 2])
raw_corpus["label"] = raw_corpus["1"].map(literal_eval)
raw_corpus.columns = ["text", "n", "label"]

# read NER file as JSON

# pattern depends on providen data format
pattern = r'\s*\|\s*'
# clean the NER pool
for entity in ner_dct:
    ner_dct[entity] = re.split(pattern, ner_dct[entity])


if __name__ == "__main__":
    
    with open('./data/raw_corpus.pkl', 'wb') as file:
        pickle.dump(raw_corpus["text"].values , file)

    with open('./data/ner.pkl', 'wb') as file:
        pickle.dump(ner_dct , file)
    
