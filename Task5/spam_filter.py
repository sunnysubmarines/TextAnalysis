import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from math import log
import pandas as pd


class Classifier:
    ham_sms_probability = {}
    spam_sms_probability = {}
    sum_tf_ham_idf = 0
    spam_probability = {}
    sum_tf_spam_idf = 0
    ham_probability = {}

    def task(dataset_path):

        df = pd.read_csv(dataset_path, encoding='latin1')

        df["class"] = df["type"].map({'ham': 0, 'spam': 1})

        # model = Vectorize.word_vectors(df["text"].to_numpy())
        # df["vectors"]=df["text"].apply(lambda x: Vectorize.vectorize_list(x, model))

        df["prep_text"] = df["text"].apply(Classifier.clean_text)
        train = df.head(int(len(df) * 0.8)).reset_index(drop=True)
        test = df.tail(int(len(df) * 0.2)).reset_index(drop=True)
        Classifier.train(train)
        test["predict"] = Classifier.predict(test["prep_text"]).values()
        Classifier.metrics(test['class'], test['predict'])
        header = ["class", "predict", "text"]
        test.to_csv("Task5/test_prediction.csv", index=False, columns=header)

    def clean_text(sms):
        porter = nltk.PorterStemmer()
        sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
        sms_no_punctuation = "".join(sms_no_punctuation).split()
        sms_no_punctuation_no_stopwords = \
            [porter.stem(word.lower()) for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]
        return sms_no_punctuation_no_stopwords

    def train(dataframe):
        sms = dataframe["prep_text"]
        labels = dataframe["class"]
        sms_count = len(sms)
        spam_count, ham_count = labels.value_counts()[1], labels.value_counts()[0]
        spam_words = 0
        ham_words = 0
        spam_tf = dict()
        ham_tf = dict()
        spam_idf = dict()
        ham_idf = dict()
        for i in range(sms_count):
            count = list()
            for word in sms[i]:
                if labels[i]:
                    spam_tf[word] = spam_tf.get(word, 0) + 1
                    spam_words += 1
                else:
                    ham_tf[word] = ham_tf.get(word, 0) + 1
                    ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if labels[i]:
                    spam_idf[word] = spam_idf.get(word, 0) + 1
                else:
                    ham_idf[word] = ham_idf.get(word, 0) + 1

        for word in spam_tf:
            Classifier.spam_probability[word] = (spam_tf[word]) * log(
                (sms_count) / (spam_idf[word] + ham_idf.get(word, 0)))
            Classifier.sum_tf_spam_idf += Classifier.spam_probability[word]
        for word in spam_tf:
            Classifier.spam_probability[word] = (Classifier.spam_probability[word] + 1) / (
                    Classifier.sum_tf_spam_idf + len(list(Classifier.spam_probability.keys())))

        for word in ham_tf:
            Classifier.ham_probability[word] = (ham_tf[word]) * log((sms_count) \
                                                                    / (spam_idf.get(word, 0) + ham_idf[word]))
            Classifier.sum_tf_ham_idf += Classifier.ham_probability[word]
        for word in ham_tf:
            Classifier.ham_probability[word] = (Classifier.ham_probability[word] + 1) / (
                    Classifier.sum_tf_ham_idf + len(list(Classifier.ham_probability.keys())))
        Classifier.spam_sms_probability, Classifier.ham_sms_probability = spam_count / sms_count, ham_count / sms_count

    def metrics(labels, predictions):

        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i in range(len(labels)):
            true_pos += int(labels[i] == 1 and predictions[i] == 1)
            true_neg += int(labels[i] == 0 and predictions[i] == 0)
            false_pos += int(labels[i] == 0 and predictions[i] == 1)
            false_neg += int(labels[i] == 1 and predictions[i] == 0)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        Fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy)

    def classify(processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in Classifier.spam_probability:
                pSpam += log(Classifier.spam_probability[word])
                pHam -= log(Classifier.sum_tf_ham_idf + len(Classifier.ham_probability.keys()))
            if word in Classifier.ham_probability:
                pHam += log(Classifier.ham_probability[word])
                pSpam -= log(Classifier.sum_tf_spam_idf + len(Classifier.spam_probability.keys()))
            pSpam += log(Classifier.spam_sms_probability)
            pHam += log(Classifier.ham_sms_probability)
        return pSpam >= pHam

    def predict(data):
        result = dict()
        for (i, message) in enumerate(data):
            result[i] = int(Classifier.classify(message))
        return result
