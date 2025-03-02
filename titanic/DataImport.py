import pandas as pd

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
GENDER_DATA_FILE_NAME = "gender_submission.csv"

AGE_CUTS = [-1, 0, 5, 12, 18, 35, 60, 100]
AGE_LABELS = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", "Adult", "Senior"]


def get_titanic_data():
    return pd.read_csv(TRAIN_FILE_NAME)


def get_titanic_test_data():
    return pd.read_csv(TEST_FILE_NAME)


def get_titanic_test_results():
    return pd.read_csv(GENDER_DATA_FILE_NAME)


def filter_age(data):
    data['Age'] = data['Age'].fillna(-0.5)
    data['Age_category'] = pd.cut(data['Age'], bins=AGE_CUTS, labels=AGE_LABELS)
    return data


def create_dummies(data, column):
    dummies = pd.get_dummies(data[column], prefix=column)
    return pd.concat([data, dummies], axis=1)


def clean_titanic_data(data):
    data = filter_age(data)
    data = create_dummies(data, 'Pclass')
    data = create_dummies(data, 'Age_category')
    data = create_dummies(data, 'Sex')
    data = create_dummies(data, 'SibSp')
    data = data.dropna()
    return data


def get_clean_train_data():
    train_data = get_titanic_data()
    return clean_titanic_data(train_data)


def get_clean_test_data():
    test_data = get_titanic_test_data()
    return clean_titanic_data(test_data)
