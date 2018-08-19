import pandas as pd

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
GENDER_DATA_FILE_NAME = "gender_submission.csv"


def get_titanic_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_FILE_NAME)


def get_titanic_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_FILE_NAME)


def get_gender_data() -> pd.DataFrame:
    return pd.read_csv(GENDER_DATA_FILE_NAME)


def filter_age(data: pd.DataFrame, cuts: list, labels: list) -> pd.DataFrame:
    data['Age'] = data['Age'].fillna(-0.5)
    data['Age_category'] = pd.cut(data['Age'], cuts, labels=labels)
    return data


def create_dummies(data: pd.DataFrame, column: str) -> pd.DataFrame:
    dummies = pd.get_dummies(data[column], prefix=column)
    return pd.concat([data, dummies], axis=1)


def clean_titanic_data(data: pd.DataFrame, age_cuts:list, age_labels: list):
    data = filter_age(data, age_cuts, age_labels)
    data = create_dummies(data, 'Pclass')
    data = create_dummies(data, 'Age_category')
    data = create_dummies(data, 'Sex')
    return data


def get_clean_train_data(age_cuts: list, age_labels: list):
    # TODO: Use builder notation so this becomes easier
    train_data = get_titanic_data()
    train_data = clean_titanic_data(train_data, age_cuts, age_labels)
    return train_data


def get_clean_test_data(age_cuts: list, age_labels: list):
    test_data = get_titanic_test_data()
    test_data = clean_titanic_data(test_data, age_cuts, age_labels)
    return test_data
