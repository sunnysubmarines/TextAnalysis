import fnmatch

import matplotlib as matplotlib

import Task1.parse
from Task1.parse import Parse
from Task2.tokenizer import Tokenizer
from Task3.dictionaries import ContrastAnalysis
from Task4.vectorize import Vectorize
from Task6.word2vec import VectorizwWord2vec
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.cluster.hierarchy import  linkage, dendrogram
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from Task5.spam_filter import Classifier


def task1():
    # path1 = "Task1/txt.xml"
    # path = "Task1/txt.txt"
    # path = "Task1/pdf.pdf"
    # path1 = "Task1/pdf.xml"
    # path = "Task1/doc.docx"
    # path1 = "Task1/doc.xml"
    path = "http://2035.media/2017/12/06/demography/"
    path1 = "Task1/html.xml"
    if fnmatch.fnmatch(path, '*.txt'):
        Parse.read_txt(path, path1)
    elif fnmatch.fnmatch(path, '*.pdf'):
        Parse.read_pdf(path, path1)
        # txt = Parse.extract_text_from_pdf(path)
    elif fnmatch.fnmatch(path, '*.docx'):
        Parse.read_docx(path, path1)
    else:
        Parse.read_html(path, path1)




# text = task1()

def task2():
    text = Tokenizer.regex_unique_tokens(Tokenizer.readtxt("Task2/text.txt"))
    l1 = text
    l2 = text
    hamming_matrix = np.zeros((len(l1), len(l2)))
    lev_matrix = np.zeros((len(l1), len(l2)))
    jaro_matrix = np.zeros((len(l1), len(l2)))

    for i in range(0, len(l1)):
        for j in range(0, len(l2)):
            hamming_matrix[i, j] = Tokenizer.hamming(l1[i], l2[j])
            lev_matrix[i, j] = Tokenizer.levenshtein(l1[i], l2[j])
            jaro_matrix[i, j] = Tokenizer.jarowinkler(l1[i], l2[j])
    plt.figure(figsize=(10, 8), dpi=300)
    plt.title("Hamming")
    c_link = shc.linkage(hamming_matrix, 'complete', 'correlation')
    dend = shc.dendrogram(c_link, 30, None, None, True, 'top', labels=l1)
    plt.savefig('hamming_distance.png')
    plt.title("Levenshtein")
    c_link = shc.linkage(lev_matrix, 'complete', 'correlation')
    dend = shc.dendrogram(c_link, 30, None, None, True, 'top', l1)
    plt.savefig('lev_distance.png')
    plt.title("Jaro-Winkler")
    c_link = shc.linkage(jaro_matrix, 'complete', 'correlation')
    dend = shc.dendrogram(c_link, 30, None, None, True, 'top', l1)
    plt.savefig('jarowinkler_distance.png')


def task3():
    folder1, folder2, folder3 = "Task3\group1", "Task3\group2", "Task3\group3"
    # ContrastAnalysis.task([folder1, folder2, folder3])
    ContrastAnalysis.task2([folder1, folder2, folder3])
    # ContrastAnalysis.task([ "Task3\group3"])


def task4():
    path = "Task4/articles.csv"
    Vectorize.task(path)


def task5():
    path = "Task5/sms_spam.csv"
    Classifier.task(path)


def task6():
    path = "Task4/articles.csv"
    VectorizwWord2vec.task(path)


task6()
