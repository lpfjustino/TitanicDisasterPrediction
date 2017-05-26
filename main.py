import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import naive_bayes as nb

def preprocess(data_frame, train_set=True):
    # Generates a copy of the original dataframe
    df = data_frame.copy()

    # Cutting out irrelevant features
    id = data_frame.loc[:, 'PassengerId']
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    data_set = data_frame.loc[:, features]
    if train_set:
        expected = data_frame.loc[:, 'Survived']
    else:
        expected = None

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


def preprocess_test_set(data_frame):
    id, data_set, _ = preprocess(data_frame, False)
    return id, data_set

def benchmark(data_set, expected):
    n_rows, n_features = data_set.shape

    train_portion = int(0.7 * n_rows)

    train_set = data_set.loc[:train_portion, :].as_matrix()
    train_labels = expected.loc[:train_portion].as_matrix()

    test_set = data_set.loc[train_portion:, :].as_matrix()
    test_labels = expected.loc[train_portion:].as_matrix()

    # gnb = GaussianNB()
    # gnb.fit(train_set, train_labels)
    # observed = gnb.predict(test_set)

    model = nb.NaiveBayes()
    model.fit(train_set, train_labels)
    observed = model.predict(test_set)

    matches = observed == test_labels
    train_size = test_set.shape[0]
    accuracy = sum(matches) / train_size
    print(accuracy)


def run(data_set, test_set):
    gnb = GaussianNB()
    gnb.fit(data_set, expected)

    observed = gnb.predict(test_set)

    print('PassengerId,Survived')
    for i, passenger in test_set.iterrows():
        print(id.loc[i], ',', observed[i], sep='')


# Reading titanic dataset
data_set = pd.read_csv("train.csv")
id, data_set, expected = preprocess(data_set)
test_set = pd.read_csv("test.csv")
id, test_set = preprocess_test_set(test_set)

benchmark(data_set, expected)
#run(data_set, test_set)