import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import isnan

from enum import Enum

class Age(Enum):
    INFANT = 0
    CHILD = 1
    TEEN = 2
    ADULT = 3
    ELDERLY = 4


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
    disc_ages = pd.Series(np.digitize(ages, bins))
    data_set.loc[:, 'Age'] = disc_ages

    return id, data_set, expected


# Reading titanic dataset
data_set = pd.read_csv("train.csv")

preprocess(data_set)