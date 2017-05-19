import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time

from sklearn.naive_bayes import GaussianNB

from enum import Enum

class Age(Enum):
    infant = 0
    child = 1
    teen = 2
    adult = 3
    elderly = 4


def preprocess(data_frame):
    # Generates a copy of the original dataframe
    df = data_frame.copy()

    # Cutting out irrelevant features
    id = data_frame.loc[:, 'PassengerId']
    expected = data_frame.loc[:, 'Survived']
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    data_set = data_frame.loc[:, features]

    # Replace empty ages by their means by sex and passenger class groups
    # Average age by sex and passenger class
    avg_ages = data_set.groupby(['Sex', 'Pclass']).mean()['Age']
    ages = data_set.loc[:, 'Age']
    missing_age = data_set.loc[ages.isnull() == True, :]
    for i, p in missing_age.iterrows():
        avg = int(avg_ages[p.Sex][p.Pclass])
        data_set.set_value(i,'Age', avg)

    # Discretization of ages
    bins = [2, 10, 18, 60]
    #disc_ages = pd.Series(np.digitize(ages, bins)).map(lambda x: Age(x).name)
    disc_ages = pd.Series(np.digitize(ages, bins))
    data_set.loc[:, 'Age'] = disc_ages

    # Discretization of fares
    fares = data_set.loc[:, 'Fare']
    bins = np.linspace(0, 100, 5)
    disc_fares = pd.Series(np.digitize(fares, bins))
    data_set.loc[:, 'Fare'] = disc_fares

    # Replacing gender
    data_set.replace(['male','female'], [0,1], inplace=True)

    # Replacing embarkation port and filling unknown ones with most frequent embark port
    data_set.replace(['C', 'Q', 'S'], [0,1,2], inplace=True)
    most_freq_port = data_set.loc[:,'Embarked'].max()
    is_missing = data_set.isnull().values.sum(axis=1)
    missing_emb = data_set.loc[is_missing == 1, :]
    for i, miss in missing_emb.iterrows():
        data_set.set_value(i,'Embarked', most_freq_port)

    return id, data_set, expected


# Reading titanic dataset
data_set = pd.read_csv("train.csv")


id, data_set, expected = preprocess(data_set)

n_rows, n_features = data_set.shape

train_portion = int(0.7 * n_rows)

train_set = data_set.loc[:train_portion, :]
train_labels = expected.loc[:train_portion]

test_set = data_set.loc[train_portion:, :]
test_labels = expected.loc[train_portion:].as_matrix()

gnb = GaussianNB()
gnb.fit(train_set, train_labels)

observed = gnb.predict(test_set)
accuracy = sum(observed == test_labels) / len(observed)
print(accuracy)


# for i, example in test_set.iterrows():
#     observed = gnb.predict(example)
#     print(observed, ' vs ', test_labels[i])
#     time.sleep(2)