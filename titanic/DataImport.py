import pandas as pd

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
GENDER_DATA_FILE_NAME = "gender_submission.csv"


def get_titanic_train_x_data() -> pd.DataFrame:
    return get_titanic_data().drop('Survived', axis=1)


def get_titanic_train_y_data() -> pd.DataFrame:
    return get_titanic_data()['Survived']


def get_titanic_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_FILE_NAME)


def get_titanic_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_FILE_NAME)


def get_gender_data() -> pd.DataFrame:
    return pd.read_csv(GENDER_DATA_FILE_NAME)
