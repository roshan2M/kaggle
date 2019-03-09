import numpy as np
import pandas as pd
import DataImport as di
import Vectorization as vc

from sklearn.model_selection import StratifiedShuffleSplit


def split_data(toxic_comments_train: pd.DataFrame):
    stratified_sample = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_index, test_index = next(stratified_sample.split(np.zeros(len(toxic_comments_train)),
                                                           toxic_comments_train[di.CATEGORIES].sum(axis=1)))
    return train_index, test_index


def get_train_test_comment_vectors():
    toxic_comments_train_set = di.load_train_data()
    comment_vectors = vc.get_comment_vectors(toxic_comments_train_set)
    train_idx, test_idx = split_data(toxic_comments_train_set)

    train_vectors = comment_vectors[train_idx]
    test_vectors = comment_vectors[test_idx]
    return train_vectors, test_vectors


def get_train_test_comment_classes():
    toxic_comments_train_set = di.load_train_data()
    train_idx, test_idx = split_data(toxic_comments_train_set)

    train_classes = toxic_comments_train_set[di.CATEGORIES].loc[train_idx]
    test_classes = toxic_comments_train_set[di.CATEGORIES].loc[test_idx]
    return train_classes, test_classes


def get_evaluation_vectors():
    toxic_comments_test_set = di.load_test_data()
    return vc.get_comment_vectors(toxic_comments_test_set)


def get_testing_set_ids():
    toxic_comments_test_set = di.load_test_data()
    return toxic_comments_test_set['id']
