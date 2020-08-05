#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:25:28 2020

@author: akashbudholia
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv('/Users/akashbudholia/Downloads/news.csv')

x_train, x_test, y_train, y_test = train_test_split(dataset['text'], labels, test_size = 0.2, random_state = 7)


# TFIDF vectorizer:

tfidf_vectorizer = TfidfVectorizer(stop_words= 'english', max_df = 0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)

tfidf_test = tfidf_vectorizer.transform(x_test)


# Fitting a PassiveAgressive Classifier:  Predicting the test_score and calculate the accuracy score

pac = PassiveAggressiveClassifier(max_iter = 50)

pac.fit(tfidf_train, y_train)



y_pred = pac.predict(tfidf_test)

score = accuracy_score(y_test, y_pred)




