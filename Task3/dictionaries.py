import collections
import math
import os


from suffix_trees import STree
from Task2.tokenizer import Tokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class ContrastAnalysis:
    stopwords_path = "Task3/stopwords.txt"
    topics_path = "Task3/topic_dictionaries.txt"

    def docsList(folderPath):
        files = []
        docs = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(folderPath):
            for file in f:
                if '.txt' in file:
                    files.append(os.path.join(r, file))

        for f in files:
            tmp = open(f.replace('\\', '/'), "r", encoding='utf-8').read()
            docs.append(tmp)
        lower_docs = []
        for i in docs:
            lower_docs.append(Tokenizer.regex_tokens(i.lower()))
        return lower_docs

    def task(paths):
        docs = []
        for path in paths:
            docs.append(ContrastAnalysis.docsList(path))
        docs = Tokenizer.flatten(docs)
        tf_idfs = ContrastAnalysis.tfidf(docs)
        stopwords = []
        for i in tf_idfs:
            for word, val in i.items():
                if val < 0.00008:
                    stopwords.append(word)
        stopwords = Tokenizer.unique(stopwords)
        with open(ContrastAnalysis.stopwords_path, 'w') as filehandle:
            for listitem in stopwords:
                filehandle.write('%s ' % listitem)
        print(stopwords)

    def task2(paths):
        stopwords = open(ContrastAnalysis.stopwords_path, "r", encoding='utf-8').read()
        docs = []

        for path in paths:
            docs.append(ContrastAnalysis.docsList(path))
        docs = Tokenizer.flatten(docs)
        filtered_docs = []
        lemmatizer = WordNetLemmatizer()
        for doc in docs:
            filtered_sentence = []
            for w in doc:
                if w not in stopwords and len(w) > 2:
                    filtered_sentence.append(lemmatizer.lemmatize(w))
            filtered_docs.append(filtered_sentence)
        tf_idfs = ContrastAnalysis.tfidf(filtered_docs)
        tf_idf_sorted = []

        # group1= tf_idfs[0:10]
        # group1 = sorted(group.items(), key=lambda item: item[1], reverse=True))
        group2 = tf_idfs[10:19]
        print(len(tf_idfs))
        print(len(tf_idfs[0]))

        for i in tf_idfs:
            tf_idf_sorted.append(sorted(i.items(), key=lambda item: item[1], reverse=True))
        group1, group2 = [], []
        for i in range(0, 10):
            group1 = group1 + tf_idf_sorted[i]
            group2 = group2 + tf_idf_sorted[i + 10]
        group1 = sorted(group1, key=lambda item: item[1], reverse=True)
        group2 = sorted(group2, key=lambda item: item[1], reverse=True)
        group1_selected = [i[0] for i in group1]
        group1_selected = [x for i, x in enumerate(group1) if i == group1.index(x)][0:10]
        group2_selected = [i[0] for i in group2]
        group2_selected = [x for i, x in enumerate(group2) if i == group2.index(x)][10:20]

        print(tf_idf_sorted)
        N = 10
        with open(ContrastAnalysis.topics_path, "a") as file_object:
            for i in group2_selected:
                file_object.write(i[0] + "\n")

    def tf(self):
        # tfList = []
        tf_dict = {}
        unique = [x for i, x in enumerate(self) if i == self.index(x)]
        tf_dict = dict.fromkeys(unique, 0)
        tree = STree.STree(self)
        for i in unique:
            tf_dict[i] = len(tree.find_all(i))
        for i in tf_dict:
            tf_dict[i] = tf_dict[i] / float(len(self))
        return tf_dict

    def idf(word, corpus):
        docs_count = len(corpus)

        N = sum([1.0 for i in corpus if word in i])
        return math.log10(docs_count / N)

    def tfidf(corpus):
        documents_list = []
        stopwords = []
        for text in corpus:
            tf_idf_dictionary = {}
            computed_tf = ContrastAnalysis.tf(text)
            for word in computed_tf:
                tf_idf_dictionary[word] = computed_tf[word] * ContrastAnalysis.idf(word, corpus)
            documents_list.append(tf_idf_dictionary)
        return documents_list
