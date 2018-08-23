
# coding: utf-8

# In[96]:


import os
import json
import spacy

import numpy as np

from time import time

import re
import os
import codecs
import spacy
import sklearn
import matplotlib.pyplot as plt
from sklearn import model_selection
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

nlp_spacy_en = None
nlp_spacy_es = None

def get_spacy_model(lang="en"):
    global nlp_spacy_en
    global nlp_spacy_es
    if lang == "en":
        if nlp_spacy_en is None: 
            nlp_spacy_en = spacy.load(lang)
        return nlp_spacy_en
    elif lang == "es":
        if nlp_spacy_es is None: 
            nlp_spacy_es = spacy.load(lang)
        return nlp_spacy_es


# In[97]:


class Dataset():
    
    def __init__(self, dataset_path, n_splits=3, ratio=0.3, augment=False):
        self.dataset_path = dataset_path
        self.augment = augment
        self.n_splits = n_splits
        self.ratio = ratio
        self.X, self.y = self.load()
        self.splits = self.stratified_split(self.X, self.y, self.n_splits, self.ratio)
    
    def load(self):
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)
            X = [sample["text"] for sample in dataset["sentences"]]
            y = [sample["intent"] for sample in dataset["sentences"]]
        return X, y
    
    def stratified_split(self, X, y, n_splits=10, test_size=0.2):
        skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        skf.get_n_splits(X, y)
        splits = []
        for train_index, test_index in skf.split(X, y):
            # print("TRAIN:", train_index, "\n\n", "TEST:", test_index, "\n\n")
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            # add augmentation code here
            splits.append({"train": {"X": X_train, "y": y_train},
                           "test": {"X": X_test, "y": y_test}})
        return splits
    
    def get_splits(self):
        return self.splits 


# In[98]:


dataset = Dataset("/home/dash/projects/imli/data/datasets/AskUbuntuCorpus.json")
splits = dataset.get_splits()
for split in splits:
    print("X train", split["train"]["X"][: 2])
    print("y train", split["train"]["y"][:2])
    print("X test", split["test"]["X"][: 2])
    print("y test", split["test"]["y"][:2])


# In[99]:


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens

class SemhashFeaturizer:
    def __init__(self):
        self.vectorizer = self.get_vectorizer()
    
    def get_vectorizer(self):
        return TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False,
                               sublinear_tf=True, tokenizer=semhash_tokenizer)
    
    def fit(self, X, *args, **kwargs):
        self.vectorizer.fit(X)
        
    
    def transform(self, X):
        return self.vectorizer.transform(X).toarray()


# In[100]:


X, y = ["hello", "I am a boy"], ["A", "B"]

semhash_featurizer = SemhashFeaturizer()
semhash_featurizer.fit(X, y)
X_ = semhash_featurizer.transform(X)
#print(X_)


# In[101]:


class W2VFeaturizer:
    def __init__(self, lang):
        self.lang = lang
    
    def fit(self, X, *args, **kwargs):
        pass
    
    def transform(self, x):
        return np.array([get_spacy_model(self.lang)(s).vector for s in x])


# In[102]:


X, y = ["hello", "I am a boy"], ["A", "B"]
glove_path = ""
w2v_featurizer = W2VFeaturizer("en")
w2v_featurizer.fit(X, y)
X_ = w2v_featurizer.transform(X)
#print(X_)


# In[106]:


class Trainer:
    def __init__(self, splits, featurizer, path="data/plots", lang="en", name="default"):
        self.path = os.path.join(path, name)
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.splits = splits
        self.featurizer = featurizer
        self.lang = lang
        self.results = None
    
    def get_X_andy_from_split(self, split):
        train_corpus, y_train = split["train"]["X"], split["train"]["y"]
        test_corpus, y_test = split["test"]["X"], split["test"]["y"]
        self.featurizer.fit(train_corpus)
        self.featurizer.fit(test_corpus)
        X_train = self.featurizer.transform(train_corpus)
        X_test = self.featurizer.transform(test_corpus)
        return X_train, y_train, X_test, y_test
    
    def train(self):
        
        parameters_mlp={'hidden_layer_sizes':[(100,50),(300,100,50),(200,100)]}
        parameters_RF={ "n_estimators" : [50,60,70], "min_samples_leaf" : [1, 2]}
        k_range = list(range(1, 11))
        parameters_knn = {'n_neighbors':k_range}
        
        for i_s, split in enumerate(self.splits):
            print("Evaluating Split {}".format(i_s))
            X_train, y_train, X_test, y_test = self.get_X_andy_from_split(split)
            print("Train Size: {}\nTest Size: {}".format(X_train.shape[0], X_test.shape[0]))
            results = []
            #alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
            knn=KNeighborsClassifier(n_neighbors=5)
            for clf, name in [  
                    (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                    (GridSearchCV(knn,parameters_knn, cv=10),"gridsearchknn"),
                    (GridSearchCV(MLPClassifier(activation='relu'), parameters_mlp, cv=3),"gridsearchmlp"),
                    (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                    (GridSearchCV(RandomForestClassifier(n_estimators=10),parameters_RF, cv=10),"gridsearchRF")]:
                print('=' * 80)
                print(name)
                results.append(self.benchmark(clf, X_train, y_train, X_test,
                                          y_test))

            for penalty in ["l2", "l1"]:
                print('=' * 80)
                print("%s penalty" % penalty.upper())
                # Train Liblinear model
                #grid=(GridSearchCV(LinearSVC,parameters_Linearsvc, cv=10),"gridsearchSVC")
                #results.append(benchmark(LinearSVC(penalty=penalty), X_train, y_train, X_test, y_test, target_names,
                                        # feature_names=feature_names))
                results.append(self.benchmark(LinearSVC(penalty=penalty,
                                                    dual=False,tol=1e-3),
                                         X_train, y_train, X_test, y_test, target_names))

                # Train SGD model
                results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                                       penalty=penalty),
                                         X_train, y_train, X_test, y_test, target_names))

            # Train SGD with Elastic Net penalty
            print('=' * 80)
            print("Elastic-Net penalty")
            results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                                   penalty="elasticnet"),
                                     X_train, y_train, X_test, y_test, target_names))

            # Train NearestCentroid without threshold
            print('=' * 80)
            print("NearestCentroid (aka Rocchio classifier)")
            results.append(self.benchmark(NearestCentroid(),
                                     X_train, y_train, X_test, y_test, target_names))

            # Train sparse Naive Bayes classifiers
            print('=' * 80)
            print("Naive Bayes")
            results.append(self.benchmark(MultinomialNB(alpha=.01),
                                     X_train, y_train, X_test, y_test, target_names))
            results.append(self.benchmark(BernoulliNB(alpha=.01),
                                     X_train, y_train, X_test, y_test, target_names))

            print('=' * 80)
            print("LinearSVC with L1-based feature selection")
            # The smaller C, the stronger the regularization.
            # The more regularization, the more sparsity.


            results.append(self.benchmark(Pipeline([
                                          ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                                                          tol=1e-3))),
                                          ('classification', LinearSVC(penalty="l2"))]),
                                     X_train, y_train, X_test, y_test, target_names))
           # print(grid.grid_scores_)
           #KMeans clustering algorithm 
            print('=' * 80)
            print("KMeans")
            results.append(self.benchmark(KMeans(n_clusters=2, init='k-means++', max_iter=300,
                        verbose=0, random_state=0, tol=1e-4),
                                     X_train, y_train, X_test, y_test, target_names))



            print('=' * 80)
            print("LogisticRegression")
            #kfold = model_selection.KFold(n_splits=2, random_state=0)
            #model = LinearDiscriminantAnalysis()
            results.append(benchmark(LogisticRegression(C=1.0, class_weight=None, dual=False,
                  fit_intercept=True, intercept_scaling=1, max_iter=100,
                  multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                  solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
                                     X_train, y_train, X_test, y_test, target_names))

            plot_results(results)
    
    
    def benchmark(self, clf, X_train, y_train, X_test, y_test,
              print_report=True, print_top10=False,
              print_cm=True):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
        #print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))
            print()

        if print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred))

        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time
    
    def plot_results(results):
        # make some plots
        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                 color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()


# In[107]:


semhash_featurizer = SemhashFeaturizer()
dataset = Dataset("/home/dash/projects/imli/data/datasets/AskUbuntuCorpus.json")
splits = dataset.get_splits()

trainer = Trainer(splits, semhash_featurizer, lang="en", path="/home/dash/projects/imli/data/plots", 
                  name="Ubuntu")

trainer.train()
