import collections
import itertools
import os
import string

import numpy as np
import pandas as pd
from Task2.tokenizer import Tokenizer
from nltk import ngrams
from Task3.dictionaries import ContrastAnalysis
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report


class LanguageDetect:

    @staticmethod
    def task(folder_path):
        n = 2
        ua_grams = LanguageDetect.ngrams(LanguageDetect.tokenize(open("Task7/ua/1.txt", "r", encoding='utf-8').read()),
                                         n)
        ru_grams = LanguageDetect.ngrams(LanguageDetect.tokenize(open("Task7/ru/1.txt", "r", encoding='utf-8').read()),
                                         n)
        be_grams = LanguageDetect.ngrams(LanguageDetect.tokenize(open("Task7/be/1.txt", "r", encoding='utf-8').read()),
                                         n)
        ngrams = [ua_grams, ru_grams, be_grams]
        languages = ["ua", "ru", "be"]
        ngrams_freq = {}
        for i in range(0, 3):
            ngrams_freq[languages[i]] = LanguageDetect.ngram_frequency(ngrams[i])
        print(ngrams_freq)
        test_df = pd.read_csv("Task7/test.csv")
        test_df_copy = test_df.copy()
        predict_output = []
        texts = test_df['text'].tolist()
        for i in texts:
            current_model = LanguageDetect.ngrams(LanguageDetect.tokenize(i), n)
            current_model_frequency = LanguageDetect.ngram_frequency(current_model)
            score = LanguageDetect.score(ngrams_freq, current_model_frequency)
            predicted_language = [k for k in score if score[k] == max(score.values())][0]
            predict_output.append(predicted_language)
        print(test_df_copy['lang'])
        print(predict_output)
        test_df_copy['label'] = test_df['lang'].map({'ua': 0, 'ru': 1, 'be': 2}).astype(int)
        test_df.insert(loc=2, column="medium", value=predict_output, allow_duplicates=True)
        test_df_copy['label_test'] = test_df['medium'].map({'ua': 0, 'ru': 1, 'be': 2}).astype(int)
        test_df.to_csv("Task7/res.csv", index=False)
        print(classification_report(test_df_copy['label'], test_df_copy['label_test'], target_names=languages))

    @staticmethod
    def build_model(file_path, n):
        txt = open(file_path, 'r', encoding='utf-8').read()
        sents = [LanguageDetect.tokenize(sent) for sent in sent_tokenize(txt, 'russian')]
        grams = []
        for text in sents:
            grams.append(LanguageDetect.ngrams(text, n))
        grams = [item for sublist in grams for item in sublist]
        return grams

    def ngram_frequency(ngrams):
        series = pd.Series()
        for ngram in ngrams:
            if ngram not in series:
                series[ngram] = 1
            else:
                series[ngram] += 1
        return series.sort_values(ascending=False)

    def tokenize(text):
        tokens = list(text)
        # tokens = word_tokenize(text)
        tokens = [i.lower() for i in tokens if (i not in string.punctuation)]
        stop_words = stopwords.words('russian')
        stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...'])
        tokens = [i.lower() for i in tokens if (i not in stop_words)]

        # cleaning words
        tokens = [i.replace("«", "").replace(" ", "").replace("»", "") for i in tokens]
        tokens = list(' '.join(tokens).replace(" ", ""))
        return tokens

    def ngrams(words, n):
        ngrams = []
        d = collections.deque(maxlen=n)
        d.extend(words[:n])
        words = words[n:]
        for window, word in zip(itertools.cycle((d,)), words):
            ngrams.append(' '.join(window))
            d.append(word)
        return ngrams

    def distance(pretrained_series, current_series):
        distance = 0
        for ngram in current_series.index:
            if ngram not in pretrained_series.index:
                distance += abs(current_series.size - current_series[ngram])
            else:
                distance += abs(pretrained_series[ngram] - current_series[ngram])
        return distance

    def score(ngram_model, add_text, n=3):
        res = {}
        total_distance = 0
        for lang, model in ngram_model.items():
            distance = LanguageDetect.distance(model, add_text)
            res[lang] = distance
            total_distance += distance
        for item in res.keys():
            res[item] = abs(res[item] - int(total_distance / n))
        return res
