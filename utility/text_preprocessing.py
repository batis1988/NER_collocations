#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:02:44 2024

@author: antonskvarskij
"""
from typing import List, Optional
from collections import defaultdict
import re
import string
import numpy as np
import nltk
from utility.similarities import VectorSimilarities

stopwords = nltk.corpus.stopwords.words('russian') # Optional, language choice depends on you
sent_tokenizer = nltk.tokenize.sent_tokenize
word_tokenizer = nltk.tokenize.word_tokenize
lemmatizer = nltk.stem.WordNetLemmatizer()

class TextPreprocesser:
    def __init__(self, 
                 raw_corpus = List[str],
                 n_grams: int = 3,  
                 stopwords = stopwords,
                 emails: bool = False, 
                 links: bool = False):
        self.corpus = []
        self.n_grams = n_grams
        self.emails = emails
        self.links = links
        self.pattern_email = r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*'
        self.pattern_links = r'http://\S+|https://\S+'
        self.pattern_words = r'[\d+{0}]'.format(string.punctuation)
        self.pattern_ngrams = r'([^\s\w]|_)+'
        self.stopwords = stopwords
        self._sentence_tokenize(raw_corpus)
        self.corpus = self._cleansing(self.corpus)
        self.clean_corpus = [" ".join(self._word_tokenize(sent)) for sent in self.corpus]
        self.clean_corpus = list(set([sent for sent in self.clean_corpus if sent != '']))
        self.word2idx, self.idx2word, self.counter = self._return_dicts(self.clean_corpus)
        self.ngrams = [self._ngram_extractor(sent) for sent in self.clean_corpus]
        
    def _sentence_tokenize(self, corpus):
        for item in corpus:
            self.corpus += [sent for sent in sent_tokenizer(item)]
                    
    
    def _cleansing(self, text):
        if not self.emails:
            text = [re.sub(self.pattern_email, '', sent) \
                             for sent in text]
        if not self.links:
            text = [re.sub(self.pattern_links, '', sent) \
                    for sent in text]
                
        text = [re.sub(self.pattern_words, '', sent) for sent in text]
        text = [sent.lower() for sent in text]
        return text
    
    def _ngram_extractor(self, sentence):
        sent = []
        tokens = re.sub(self.pattern_ngrams, ' ', sentence).split()
        for gram in range(len(tokens) - self.n_grams + 1):
            sent.append(tokens[gram: gram + self.n_grams])
        return sent                
    
    def _word_tokenize(self, sentence):
        return [lemmatizer.lemmatize(word) for word in word_tokenizer(sentence)\
                if word not in stopwords]
    
    def _return_dicts(self, corpus):
        idx = 0
        word2idx = {}
        counter = {}
        for doc in corpus:
            words = word_tokenizer(doc)
            for word in words:
                # word2idx fullfilment 
                if word not in word2idx:
                    word2idx[word] = idx
                    idx += 1
                    counter[word] = 1
                else:
                    counter[word] += 1
        idx2word = {value: key for key, value in word2idx.items()}
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return word2idx, idx2word, counter


class NGramCheck:
    
    def __init__(self, corpus: List[List[str]], 
                 ner: str, 
                 model) -> None:
        self.model = model
        self.corpus = corpus
        self.ner = ner
        self.score = VectorSimilarities()
        self.sim_score = defaultdict(int)
        self.target = self._get_vector(self.ner)
        self.is_fitted = False
        
    def _get_vector(self, instance: str) -> np.ndarray:
        if hasattr(self.model, "get_bert_vectors"):
            target = self.model.get_bert_vectors(instance)
        else:
            target = self.model.get_word_vector(instance)
        return target
                    
    def _get_ngram_vectors(self, ngram: List[str]) -> List[np.ndarray]:
        if len(ngram) > 1:
            unit_vec = self._get_vector(' '.join(ngram))
        else: 
            unit_vec = self._get_vector(ngram)
        return unit_vec
    
    def _get_similarity(self, vector: np.ndarray) -> float: 
        return self.score.cosine_similarity(self.target, vector)
    
    def get_collocations(self, threshold=Optional[float]):
        for num, sent in enumerate(self.corpus):
            scores = []
            for ngram in sent:
                unit_vec = self._get_ngram_vectors(ngram)
                scores.append(self._get_similarity(unit_vec))
            dct = {}
            for idx in range(len(sent)):
                dct[' '.join(sent[idx])] = scores[idx]
            filtered = {key: value for key, value in dct.items() if value >= threshold}
            self.sim_score[num] = filtered
        return self.sim_score