import re
import numpy as np
import pandas as pd
import nltk
import textdistance




class Tokenizer:
    def readtxt(inputPath):
        file = open(inputPath, "r")
        lines = file.readlines()
        file.close()
        list = [x for x in lines if x is not "\n"]
        return list

    def unique(l):
        n = []
        for i in l:
            if i not in n:
                n.append(i)
        return n

    def regex_tokens(text):
        pattern = "([a-zA-Z]+)"
        string_text = "".join(text)
        result = re.findall(pattern, string_text)
        tokens = nltk.regexp_tokenize(string_text, pattern)
        return (tokens)

    def flatten(self):
        return [y for x in self for y in x]
    def regex_unique_tokens(text):
        pattern = "([a-zA-Z]+)"
        stringtext = "".join(text)
        result = re.findall(pattern, stringtext)
        tokens = nltk.regexp_tokenize(stringtext, pattern)
        return Tokenizer.unique(tokens)

    def listToTuples(list):
        tuples = []
        for i in list:
            for j in list:
                if i!=j:
                    tuples.append((i,j))
        return (tuples)

    def hamming (string1, string2):
        hamming = textdistance.Hamming()
        d = textdistance.hamming.normalized_distance(string1, string2)
        return d
    def levenshtein (string1, string2):
        lev = textdistance.Levenshtein()
        d = lev.distance(string1, string2)
        return d
    def jarowinkler (string1, string2):
        jaro = textdistance.JaroWinkler()
        d = jaro.normalized_distance(string1, string2)
        return d

