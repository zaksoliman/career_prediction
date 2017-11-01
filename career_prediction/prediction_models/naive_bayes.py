import pandas as pd
import numpy as np
import re
import string
import os
import json
from bidict import bidict
from helpers.preprocessing import preprocess_job_title_sequences
import pickle
import random
from math import ceil
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import itertools

def build_data(type="train"):

    data = preprocess_job_title_sequences(offset=False, save=False)
    examples = data[type + "_data"]
    title_to_id = bidict(data["title_to_id"])
    vocab_size = len(title_to_id)
    targets = np.zeros(len(examples))
    X = np.zeros((len(examples), vocab_size))

    for i, ex in enumerate(examples):
        targets[i] = ex[-1]
        for elem in ex[:-1]:
            X[i][elem] += 1

    return X, targets, title_to_id

def multiomial_nb():

    X_train, train_targets, titles_to_id = build_data(type="train")
    X_test, test_targets, _ = build_data(type="test")

    # Train
    multi_nb = MultinomialNB()
    print("Training Multinomial Naive Bayes...")
    multi_nb.fit(X_train, train_targets)

    # Test
    print("Running trained model on test dataset")
    predicted = multi_nb.predict(X_test)
    acc = np.mean(predicted == test_targets)

    print("Model Accuracy: " + str(acc))

if __name__ == "__main__":
    multiomial_nb()