from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import gensim
from gensim.models import Word2Vec
import pandas as pd
from Task2.tokenizer import Tokenizer
from Task3.dictionaries import ContrastAnalysis
from Task4.vectorize import Vectorize
import numpy as np


class VectorizwWord2vec:
    path_wordsim = 'Task4/compare.csv'

    def task(dataset_path):
        df = pd.read_csv(dataset_path)
        texts = ' '.join(df['text'])
        add_data = ' '.join(pd.read_csv("Task4/BBC news dataset.csv")["description"])

        prepared_sentences = Vectorize.preprocessing(texts)
        prepared_sentences2 = Tokenizer.flatten(Vectorize.preprocessing(add_data))
        sentences = prepared_sentences + prepared_sentences2

        cbow_model = gensim.models.Word2Vec(sentences, min_count=1, size=100, window=5)
        skip_gram_model = gensim.models.Word2Vec(sentences, min_count=1, size=100, window=5, sg=1)
        wordsim = pd.DataFrame(pd.read_csv("Task4/compare.csv"))
        list1 = wordsim['word1'].tolist()
        list2 = wordsim['word2'].tolist()
        cbow_sim = []
        skip_gram_sim = []
        for w1, w2 in zip(list1, list2):
            cbow_sim.append(cbow_model.similarity(w1, w2))
            skip_gram_sim.append(skip_gram_model.similarity(w1, w2))
        wordsim = wordsim.drop("Dkl", 1)
        wordsim = wordsim.drop("KL", 1)
        wordsim.insert(loc=6, column="v2v cbow", value=cbow_sim, allow_duplicates=True)
        wordsim.insert(loc=7, column="v2v skip_gram", value=skip_gram_sim, allow_duplicates=True)
        wordsim = wordsim.round({"Jac": 5, "DKL": 5, "v2v cbow": 5, "v2v skip_gram": 5})
        print(np.corrcoef(wordsim['human(mean)'].tolist(), wordsim['v2v cbow'].tolist())[0, 1])
        print(np.corrcoef(wordsim['human(mean)'].tolist(), wordsim['v2v skip_gram'].tolist())[0, 1])

        wordsim.to_csv("Task6/compare.csv", index=False)
