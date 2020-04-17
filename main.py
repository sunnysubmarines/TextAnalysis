import fnmatch

import matplotlib as matplotlib

import Task1.parse
from Task1.parse import Parse
from Task2.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.cluster.hierarchy import  linkage, dendrogram
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from Task3.articles_test import ContrastAnalysis

def task1():
    #path1 = "Task1/txt.xml"
    #path = "Task1/txt.txt"
    # path = "Task1/pdf.pdf"
    # path1 = "Task1/pdf.xml"
    # path = "Task1/doc.docx"
    # path1 = "Task1/doc.xml"
    path = "http://2035.media/2017/12/06/demography/"
    path1 = "Task1/html.xml"
    if fnmatch.fnmatch(path, '*.txt'):
        Parse.readtxt(path, path1)
    elif fnmatch.fnmatch(path, '*.pdf'):
        Parse.readPdf(path, path1)
        # txt = Parse.extract_text_from_pdf(path)
    elif fnmatch.fnmatch(path, '*.docx'):
        Parse.readDocx(path, path1)
    else:
        Parse.readHtml(path, path1)




# text = task1()

def task2():
    text = Tokenizer.regexTokens(Parse.readtxt("Task2/text.txt"))
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
    plt.figure
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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram = shc.dendrogram(linkage_matrix, **kwargs)
    plt.show()


# tuples = Tokenizer.listToTuples(Tokenizer.regexTokens(text))
# hamming = []
# t= tuples[0]
# for t in tuples:
#     t = t+(Tokenizer.hamming(t[0], t[1]),Tokenizer.jarowinkler(t[0],t[1]),)
#     print(t)
# print(tuples)
# X = np.random.rand(15, 10) # 15 samples, with 10 dimensions each
# fig = ff.create_dendrogram(X, 1.5)
# fig.update_layout(800, 500)
# fig.show()


def task3():
    f = open("Task3/group 1/1.txt", "r", encoding='utf-8')
    ContrastAnalysis.task(f.read())

#task1()
#task2()
task3()