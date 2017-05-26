import numpy as np
import pandas as pd
from pandas import DataFrame

import time

class NaiveBayes:
    data_set = []

    def naive(self, query):
        n_rows, n_cols = self.data_set.shape
        classes = np.unique(self.data_set[:,n_cols-1])
        n_classes = len(classes)
        classId = n_cols-1

        P = np.zeros(n_classes)
        for i in range(n_classes):
            class_count = self.data_set[:, classId] == classes[i]
            P_i = sum(class_count)/n_rows

            prod = 1

            for j in range(len(query)):
                same_attribute_j = self.data_set[:,j] == query[j]
                same_class = self.data_set[:,classId] == classes[i]
                P_hip_given_ci = sum(same_attribute_j & same_class)/sum(same_class)
                prod *= P_hip_given_ci

            prod *= P_i
            P[i] = prod

        if sum(P) != 0:
            P = P/sum(P)

        classification = np.argmax(P)
        return classification


    def fit(self, train_set, train_labels):
        self.data_set = np.c_[train_set, train_labels]

    def predict(self, test_set):
        classifications = []

        for i, example in enumerate(test_set):
            print(i)
            classification = self.naive(example)
            classifications.append(classification)

        return classification

# data_set = pd.read_table("tenis.dat").as_matrix()
# nb = NaiveBayes()
# classes = data_set[:,-1]
# attributes = data_set[:,:-1]
# nb.fit(classes, attributes)
# nb.predict(attributes)
# nb.naive(attributes[0])