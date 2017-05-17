import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def preprocess(data):
    print(data['Fare'])

    # Cutting out irrelevant features
    id = data['PassengerId']
    expected = data['Survived']
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

    data_set = data[features]

    return id, data_set, expected


# Reading titanic dataset
data_set = pd.read_csv("train.csv")

preprocess(data_set)