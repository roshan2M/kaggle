import pandas as pd

TRAIN_TOXIC_COMMENTS_FILE = '../data/train.csv'
TEST_TOXIC_COMMENTS_FILE = '../data/test.csv'

CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat',
              'insult', 'identity_hate']
TEXT_COLUMN = 'comment_text'


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_TOXIC_COMMENTS_FILE)


def load_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_TOXIC_COMMENTS_FILE)
