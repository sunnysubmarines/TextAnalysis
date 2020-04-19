import re
import numpy as np
import pandas as pd
import nltk
import textdistance




class Tokenizer:
    def unique(l):
        n = []
        for i in l:
            if i not in n:
                n.append(i)
        return n

    def regexTokens(text):
        pattern = "([a-zA-Z]+)"
        stringtext = "".join(text)
        result = re.findall(pattern, stringtext)
        tokens = nltk.regexp_tokenize(stringtext, pattern)
        #for match in result:
         #   print (match.groups)

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

