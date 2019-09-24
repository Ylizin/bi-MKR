#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# content-based recommendation 
# calc the similirity between libraries by using lda method.

import os
from collections import defaultdict
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models, similarities

tmp_path = '/Users/Abraham/Workspace/app-lib/lda/tmp'

def tokenize_docs(docs):
  if (os.path.exists(tmp_path + '/docstoken.npy')):
    return np.load(tmp_path + '/docstoken.npy', allow_pickle=True).tolist()
  else:
    # set stop words
    stop_words = stopwords.words('english')

    # punctuations
    punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'']

    # tokenize
    # tokenizer = RegexpTokenizer(r'\w+')
    # raw_tokens = tokenizer.tokenize(doc.lower())
    # raw_tokens = word_tokenize(doc.lower())

    texts = [
      word_tokenize(doc.lower())
      for doc in docs
    ]

    # stop_words and punctuations
    texts = [
      [w for w in text if not w in stop_words + punctuations]
      for text in texts
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
      for token in text:
          frequency[token] += 1
    texts = [
      [token for token in text if frequency[token] > 1]
      for text in texts
    ]

    # word stem
    ps = PorterStemmer()
    texts = [
      [ps.stem(w) for w in text]
      for text in texts
    ]

    np.save(tmp_path + '/docstoken.npy', np.array(texts))

    return texts

def get_dic_corpus(texts):
  if (os.path.exists(tmp_path + '/descdoc.dict')):
    dictionary = corpora.Dictionary.load(tmp_path + '/descdoc.dict')
    corpus = corpora.MmCorpus(tmp_path + '/descdoc.mm')
  else:
    dictionary = corpora.Dictionary(texts)
    dictionary.save(tmp_path + '/descdoc.dict')
    corpus=[dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(tmp_path + '/descdoc.mm', corpus)
  
  return dictionary, corpus

def get_model(corpus, dictionary):
  if (os.path.exists(tmp_path + '/descdoc.lda')):
    lda = models.LdaModel.load(tmp_path + '/descdoc.lda')
  else:
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=300)
    lda.save(tmp_path + '/descdoc.lda')

  return lda

def get_topn_sim(doc, model, corpus, dictionary):
  if (os.path.exists(tmp_path + '/descdoc.index')):
    index = similarities.MatrixSimilarity.load(tmp_path + '/descdoc.index')
  else:
    index = similarities.MatrixSimilarity(model[corpus])  # transform corpus to LSI space and index it
    index.save(tmp_path + '/descdoc.index')
  
  vec_bow = dictionary.doc2bow(doc.lower().split())
  vec_model = model[vec_bow]  # convert the query to LSI space

  sims = index[vec_model]
  sims = sorted(enumerate(sims), key=lambda item: -item[1])
  return sims


def recommend(doc, docs, n_sim):
  texts = tokenize_docs(docs)
  dictionary, corpus = get_dic_corpus(texts)
  lda = get_model(corpus, dictionary)
  sims = get_topn_sim(doc, lda, corpus, dictionary)
  return sims[:n_sim]
