from collections import Counter
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from Task2.tokenizer import Tokenizer
from scipy.special import rel_entr


class Vectorize:
    word_list = []
    tok2indx = {}
    indx2tok = {}
    ppmi_mat = []
    tfifd_mat = []

    def task(dataset_path):
        df = pd.read_csv(dataset_path)
        texts = ' '.join(df['text'])
        add_data = ' '.join(pd.read_csv("Task4/BBC news dataset.csv")["description"])

        prepared_sentences = Vectorize.preprocessing(texts)
        prepared_sentences2 = Tokenizer.flatten(Vectorize.preprocessing(add_data))
        sentences = prepared_sentences + prepared_sentences2

        Vectorize.ppmi_mat = Vectorize.ppmi_matrix(sentences)
        selected_words = Vectorize.word_list[0:30]
        distanses = Vectorize.similarity_matrix(selected_words, Vectorize.ppmi_mat)
        l = shc.linkage(distanses, method='complete', metric='seuclidean')
        # calculate full dendrogram
        plt.figure(figsize=(10, 8), dpi=300)
        plt.title('Jaccard distance')
        plt.xlabel('word')
        plt.ylabel('similarity')

        dend = shc.dendrogram(l, leaf_rotation=90., leaf_font_size=12., labels=selected_words)
        plt.savefig('Task4/jac_ppmi.png')
        wordsim = pd.DataFrame(pd.read_csv("Task4/compare.csv"))

        filterd_wordsim = wordsim[
            wordsim['word1'].isin(Vectorize.word_list) & wordsim['word2'].isin(Vectorize.word_list)]
        list1 = filterd_wordsim['word1'].tolist()
        list2 = filterd_wordsim['word2'].tolist()
        res = Vectorize.normalize(Vectorize.compare(list1, list2, Vectorize.ppmi_mat))
        filterd_wordsim.insert(loc=5, column="DKL", value=res, allow_duplicates=True)
        filterd_wordsim = filterd_wordsim.round(5)
        # filterd_wordsim.to_csv("Task4/compare.csv", index=False)

    def preprocessing(text):
        sentences = nltk.tokenize.sent_tokenize(text)
        lower_sentences = []
        for i in sentences:
            lower_sentences.append(Tokenizer.regex_tokens(i.lower()))
        stopwords_list = stopwords.words('english')
        stopwords_list.append("the")
        filtered_sentences = []
        lemmatizer = WordNetLemmatizer()
        for doc in lower_sentences:
            filtered_sentence = []
            for w in doc:
                if w not in stopwords_list and len(w) > 2:
                    filtered_sentence.append(lemmatizer.lemmatize(w))
            filtered_sentences.append(filtered_sentence)
        filtered_sentences = list(filter(None, filtered_sentences))
        filtered_sentences = [x for x in filtered_sentences if len(x) > 1]
        return filtered_sentences

    def ppmi_matrix(preprocess_text):
        unigram_counts = Counter()
        for sent in preprocess_text:
            for token in sent:
                unigram_counts[token] += 1
        Vectorize.word_list = list(unigram_counts.keys())
        Vectorize.tok2indx = {tok: indx for indx, tok in enumerate(unigram_counts.keys())}
        Vectorize.indx2tok = {indx: tok for tok, indx in Vectorize.tok2indx.items()}
        back_window = 2
        front_window = 2
        skipgram_cnt = Counter()
        for sent in preprocess_text:
            tokens = [Vectorize.tok2indx[tok] for tok in sent]
            for ii_word, word in enumerate(tokens):
                ii_context_min = max(0, ii_word - back_window)
                ii_context_max = min(len(sent) - 1, ii_word + front_window)
                ii_contexts = [
                    ii for ii in range(ii_context_min, ii_context_max + 1)
                    if ii != ii_word]
                for ii_context in ii_contexts:
                    skipgram = (tokens[ii_word], tokens[ii_context])
                    skipgram_cnt[skipgram] += 1
        row_indxs = []
        col_indxs = []
        dat_values = []
        ii = 0
        for (tok1, tok2), sg_count in skipgram_cnt.items():
            ii += 1
            row_indxs.append(tok1)
            col_indxs.append(tok2)
            dat_values.append(sg_count)
        word_word_matrix = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
        num_skipgrams = word_word_matrix.sum()

        row_indxs = []
        col_indxs = []

        ppmi_dat_values = []

        sum_over_words = np.array(word_word_matrix.sum(axis=0)).flatten()
        sum_over_contexts = np.array(word_word_matrix.sum(axis=1)).flatten()

        alpha = 0.75
        ii = 0
        for (tok_word, tok_context), sg_count in skipgram_cnt.items():
            ii += 1
            nwc = sg_count
            Pwc = nwc / num_skipgrams
            nw = sum_over_contexts[tok_word]
            Pw = nw / num_skipgrams
            nc = sum_over_words[tok_context]
            Pc = nc / num_skipgrams

            pmi = np.log2(Pwc / (Pw * Pc))
            ppmi = max(pmi, 0)

            row_indxs.append(tok_word)
            col_indxs.append(tok_context)
            ppmi_dat_values.append(ppmi)
        ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
        return ppmi_mat

    def word_vectors(preprocess_text):
        matrix = Vectorize.ppmi_matrix(preprocess_text)
        embedding_size = 50
        uu, ss, vv = linalg.svds(matrix, embedding_size)
        word_vecs = uu + vv.T
        return word_vecs

    def similarity_matrix(word_list, metrics):
        l = len(word_list)
        matrix = [[0 for i in range(0, l)] for i in range(0, l)]
        for i in range(0, l):
            for j in range(0, l):
                matrix[i, j] = Vectorize.words_similarity(word_list[i], word_list[j], metrics)
        return matrix

    def words_similarity(word1, word2, matrix):
        indx1 = Vectorize.tok2indx[word1]
        indx2 = Vectorize.tok2indx[word2]
        if isinstance(matrix, sparse.csr_matrix):
            v1 = matrix.getrow(indx1)
            v2 = matrix.getrow(indx2)
        else:
            v1 = matrix[indx1:indx1 + 1, :]
            v2 = matrix[indx2:indx2 + 1, :]
        # sim = cosine_similarity(v1, v2)
        sim = Vectorize.jaccard_similarity(v1, v2)
        # sim = sum(Vectorize.Dkl(v1, v2))
        return round(sim, 5)

    def normalize(v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        return v / norm

    def jaccard_similarity(vector1, vector2):
        vector1 = vector1.toarray()
        vector2 = vector2.toarray()
        intersec = np.intersect1d(vector1, vector2)
        union = np.union1d(vector1, vector2)
        return float(len(intersec)) / (len(union))

    def compare(list1, list2, matrix):
        sim_list = []
        for w1, w2 in zip(list1, list2):
            sim_list.append(Vectorize.words_similarity(w1, w2, matrix))
        return sim_list

    def vectorize_list(list, model):
        vector_list = []
        ind_row = []
        for item in list:
            i = 0
            indx = Vectorize.tok2indx[item]
            vector_list.append(model[indx:indx + 1, :])
            ind_row.append(i)
            i += 1
        # col_ind = np.range(0,19)
        return np.array(vector_list)

    def Dkl(vector1, vector2):
        vector1 = vector1.toarray() + 1
        vector2 = vector2.toarray() + 1

        return np.sum(np.where(vector1 != 0, vector1 * np.log(vector1 / vector2), 0))
        # return sum(rel_entr(vector1, vector2))
