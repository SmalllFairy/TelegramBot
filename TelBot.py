import numpy as np
import pandas as pd
import nltk
import re
import pickle
import json
import pymorphy2
from scipy import spatial
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class TelBOT(object):
    def __init__(self):
        # база вопросов и ответов
        self.faq = json.load(open('faq.json', "rb"))
        # векторизованная база вопросов
        self.vectorized_qst = json.load(open('vect_qst.json', "rb"))
        # эталонная модель вопросов для каждого кластера
        self.reference_base = json.load(open("reference_qst.json", "rb"))
        # kmeans model
        self.kmeans = joblib.load(open('km_model.pkl', "rb"))
        # word2vec vectorizer object
        self.w2v_model = pickle.load(open('word_vectors.pkl', "rb"))
        # lemmatizer
        self.morph = pymorphy2.MorphAnalyzer()
        # tf-idf vectorizer object
        vectorizer = joblib.load(open('tfidf.pkl', "rb"))
        self.tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    def handle_input(self, user_input):
        if len(user_input) < 8:
            return ['Пожалуйста, введите больше символов'], []
        else:
            qst_dist = self.qst_confidence(user_input)
            clst_dist = self.clst_confidence(user_input)

        output_num = 5

        # result1
        result1 = []
        output1 = []
        for i in range(0, output_num):
            doc_idx = qst_dist[i][1]
            result1 += [(self.faq[doc_idx][0], self.faq[doc_idx][1], qst_dist[i][0])]
        for x in result1:
            output1 += ['Вопрос:\n' + str(x[0]) + "\n" + 'Ответ:\n' + str(x[1]) + "\n" + 'Confidence: ' + str(
                round(x[2] * 100,2)) + '%']

        #result2
        result2 = []
        output2 = []
        for i in range(0, output_num):
            clst_id = clst_dist[i][1]
            result2 += [(clst_id, self.reference_base[clst_id][0], self.reference_base[clst_id][1], clst_dist[i][0])]

        for x in result2:
            output2 += ['Кластер: ' + str(x[0]) + '. ' + str(x[1]) + "\n" + 'Reference question: ' + str(
                x[2]) + "\n" + 'Confidence: ' + str(round(x[3] * 100,2)) + '%']

        return output1, output2

    def text_preproc(self, text):
        # apply nltk tokenization
        tokens = nltk.word_tokenize(text)
        # delete punctuation
        text_punct = []
        for word in tokens:
            s = (re.sub('[a-zA-Z]+|[\d\"\•\)\(\%\.\,\;\:\!\?\_\+\-\/\|\>\<\\\*\@\#\$\^\'\&\=\]]', '', word))
            text_punct.append(s)
        # lemmatize
        grams_exclusion = {'NUMB', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}
        lem_text = [self.morph.parse(word)[0].normal_form for word in text_punct if
                        self.morph.parse(word)[0].tag.POS not in grams_exclusion]
        # delete words with len(words)<=4
        text_preproc = [word for word in lem_text if len(word) > 4]

        return text_preproc

    def featurize_w2v(self, model, tfidf, text, vsize=300):
        vec = np.zeros(vsize).reshape((1, vsize))
        count = 0.
        for word in text:
            try:
                vec += model[word].reshape((1, vsize)) * tfidf[word]
                count += 1.
            except KeyError:
                continue
        if count != 0:
                vec /= count
        return vec[0]

    def vectorize(self, text):
        vectorized_text = []
        text_preproc = self.text_preproc(text)
        vectorized_text.append(self.featurize_w2v(self.w2v_model, self.tfidf, text_preproc))

        return vectorized_text

    def qst_confidence(self, text_input):
        dist_qst = []
        vectorized_text = self.vectorize(text_input)

        for i in range(len(self.vectorized_qst)):
            for j in range(0, len(self.vectorized_qst[i])):
                similarity = 1 - spatial.distance.cosine(vectorized_text, self.vectorized_qst[i][j][0])
                dist_qst += [(similarity, self.vectorized_qst[i][j][1], i)]
        dist_qst.sort(reverse=True)
        qst_conf = dist_qst[:5]

        return qst_conf

    def clst_confidence(self,text_input):
        dist_clst = []
        vectorized_text = self.vectorize(text_input)

        for i in range(0,len(self.kmeans.cluster_centers_)):
            similarity = 1 - spatial.distance.cosine(vectorized_text, self.kmeans.cluster_centers_[i])
            dist_clst += [(similarity, i)]
        dist_clst.sort(reverse=True)
        clst_conf = dist_clst[:5]

        return clst_conf
