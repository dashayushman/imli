import json
import math
import random

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

class Dataset():

    def __init__(self, dataset_path, n_splits=3, ratio=0.3, augment=False):
        self.dataset_path = dataset_path
        self.augment = augment
        self.n_splits = n_splits
        self.ratio = ratio
        self.X, self.y = self.load()
        self.keyboard_cartesian = {'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0},
                                   'r': {'x': 3, 'y': 0}, 't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0},
                                   'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0}, 'o': {'x': 8, 'y': 0},
                                   'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
                                   's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1},
                                   'c': {'x': 2, 'y': 2}, 'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2},
                                   'm': {'x': 5, 'y': 2}, 'j': {'x': 6, 'y': 1}, 'g': {'x': 4, 'y': 1},
                                   'h': {'x': 5, 'y': 1}, 'j': {'x': 6, 'y': 1}, 'k': {'x': 7, 'y': 1},
                                   'l': {'x': 8, 'y': 1}, 'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2}, }
        self.nearest_to_i = self.get_nearest_to_i(self.keyboard_cartesian)
        self.splits = self.stratified_split(self.X, self.y, self.n_splits, self.ratio, self.augment)


    def get_nearest_to_i(self, keyboard_cartesian):
        nearest_to_i = {}
        for i in keyboard_cartesian.keys():
            nearest_to_i[i] = []
            for j in keyboard_cartesian.keys():
                if self._euclidean_distance(i, j) < 1.2:
                    nearest_to_i[i].append(j)
        return nearest_to_i

    def _shuffle_word(self, word, cutoff=0.9):
        word = list(word.lower())
        if random.uniform(0, 1.0) > cutoff:
            loc = np.random.randint(0, len(word))
            if word[loc].isalpha():
                word[loc] = random.choice(self.nearest_to_i[word[loc]])
        return ''.join(word)

    def _euclidean_distance(self, a, b):
        X = (self.keyboard_cartesian[a]['x'] - self.keyboard_cartesian[b]['x']) ** 2
        Y = (self.keyboard_cartesian[a]['y'] - self.keyboard_cartesian[b]['y']) ** 2
        return math.sqrt(X + Y)

    def _augment_sentence(self, sentence, num_samples):
        sentences = []
        for _ in range(num_samples):
            sentences.append(' '.join([self._shuffle_word(item) for item in sentence.split(' ')]))
        sentences = list(set(sentences))
        return sentences

    def _augment_split(self, X_train, y_train, num_samples=10):
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            tmp_x = self._augment_sentence(X, num_samples)
            _ = [[Xs.append(item), ys.append(y)] for item in tmp_x]
        return Xs, ys

    def load(self):
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)
            X = [sample["text"] for sample in dataset["sentences"]]
            y = [sample["intent"] for sample in dataset["sentences"]]
        return X, y

    def stratified_split(self, X, y, n_splits=10, test_size=0.2, augment=False):
        skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        skf.get_n_splits(X, y)
        splits = []
        for train_index, test_index in skf.split(X, y):
            # print("TRAIN:", train_index, "\n\n", "TEST:", test_index, "\n\n")
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            # Augmentation code
            if augment:
                X_train, y_train = self._augment_split(X_train, y_train)
            splits.append({"train": {"X": X_train, "y": y_train},
                           "test": {"X": X_test, "y": y_test}})
        return splits

    def get_splits(self):
        return self.splits

dataset = Dataset("/Users/pondenkandath/projects/imli/data/datasets/AskUbuntuCorpus.json",
                  augment=True)
splits = dataset.get_splits()
for split in splits:
    print("X train", split["train"]["X"][: 2])
    print("y train", split["train"]["y"][:2])
    print("X test", split["test"]["X"][: 2])
    print("y test", split["test"]["y"][:2])

# print(set([dataset.shuffle_word('quick') for item in range(10000)]))

print(dataset._augment_sentence('hello world', 1000))


# def euclidean_distance(a, b):
#     X = (keyboard_cartesian[a]['x'] - keyboard_cartesian[b]['x']) ** 2
#     Y = (keyboard_cartesian[a]['y'] - keyboard_cartesian[b]['y']) ** 2
#     return math.sqrt(X + Y)
#
#
# nearest_to_i = {}
# for i in keyboard_cartesian.keys():
#     nearest_to_i[i] = []
#     for j in keyboard_cartesian.keys():
#         if euclidean_distance(i, j) < 1.2:
#             nearest_to_i[i].append(j)
#
#
# def shuffle_word(word, cutoff=0.9):
#     word = list(word.lower())
#     if random.uniform(0, 1.0) > cutoff:
#         loc = np.random.randint(0, len(word))
#         word[loc] = random.choice(nearest_to_i[word[loc]])
#     return ''.join(word)
#
#
# def augment_sentence(sentence, num_samples):
#     sentences = []
#     for _ in range(num_samples):
#         sentences.append([shuffle_word(item) for item in sentence.split(' ')])
#     return sentences