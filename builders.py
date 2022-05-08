import numpy as np

class FeaturesFromWords:
    def __init__(self):
        self.dict = {}

    def make_vocabulary(self, data):
        for document in data:
            for word in document:
                if word not in self.dict:
                    self.dict[word] = len(self.dict)

    def get_features(self, data):
        features = np.zeros((len(data), len(self.dict)))

        for i, document in enumerate(data):
            for word in document:
                if word in self.dict:
                    features[i][self.dict[word]] += 1

        return features

class FeaturesFromChars:
    def __init__(self):
        self.dict = {}

    def make_vocabulary(self, data):
        for document in data:
            for word in document:
                for letter in word:
                    if letter not in self.dict:
                        self.dict[letter] = len(self.dict)

    def get_features(self, data):
        features = np.zeros((len(data), len(self.dict)))

        for i, document in enumerate(data):
            for word in document:
                for letter in word:
                    if letter in self.dict:
                        features[i][self.dict[letter]] += 1

        return features
