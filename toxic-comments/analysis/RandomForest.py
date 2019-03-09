from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

import Evaluation as ev
import DataImport as di
import numpy as np
import pandas as pd


class ExtendedMultiOutputClassifier(MultiOutputClassifier):
    def transform(self, data):
        _output = self.predict_proba(data)
        return np.concatenate(_output, axis=1)


def build_random_forest_model():
    return RandomForestClassifier(n_estimators=len(di.CATEGORIES), class_weight='balanced')


def build_pipeline():
    rf_model = build_random_forest_model()
    moc = ExtendedMultiOutputClassifier(rf_model)
    nnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=0)
    return Pipeline([('moc_rf', moc), ('nnc', nnc)])


def train_pipeline(pipeline: Pipeline):
    train_vectors, test_vectors = ev.get_train_test_comment_vectors()
    train_classes, test_classes = ev.get_train_test_comment_classes()
    pipeline.fit(train_vectors, train_classes)
    return pipeline


def get_predictions(pipeline: Pipeline):
    testing_set_vectors = ev.get_evaluation_vectors()
    return pipeline.predict_proba(testing_set_vectors)


def save_predictions(test_predictions):
    output_df = pd.DataFrame()
    output_df['id'] = ev.get_testing_set_ids()
    output_df['toxic'] = test_predictions[:, 0]
    output_df['severe_toxic'] = test_predictions[:, 1]
    output_df['obscene'] = test_predictions[:, 2]
    output_df['threat'] = test_predictions[:, 3]
    output_df['insult'] = test_predictions[:, 4]
    output_df['identity_hate'] = test_predictions[:, 5]
    output_df.to_csv('predictions.csv')
    return output_df


classifier_pipeline = build_pipeline()
train_pipeline(classifier_pipeline)
predictions = get_predictions(classifier_pipeline)
save_predictions(predictions)

# rf_model = RandomForestClassifier(n_estimators=len(di.CATEGORIES), class_weight='balanced')
# train_vectors, test_vectors = ev.get_train_test_comment_vectors()
# train_classes, test_classes = ev.get_train_test_comment_classes()
#
# for category in di.CATEGORIES:
#     print('CLASS {0}'.format(category))
#     output = train_classes[category]
#     scores = cross_val_score(rf_model, train_vectors, output, scoring='roc_auc')
#     print('CV AUC {0}, Average AUC {1}'.format(scores, scores.mean()))
#
# moc = ExtendedMultiOutputClassifier(rf_model)
# nnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=0)
#
# clf = Pipeline([('moc_rf', moc), ('nnc', nnc)])
# clf.fit(train_vectors, train_classes)
#
# predictions = clf.predict_proba(test_vectors)
# print('Predictions:')
# print(predictions)
# diff = predictions - test_classes
# msd = np.sum(list(map(lambda x: np.dot(x, x.T), diff.values))) / float(len(diff))
# print('Test Mean Squared Error: {0}'.format(msd))
# print('Test Accuracy: {0}'.format(1 - msd))
#
# testing_set_vectors = ev.get_evaluation_vectors()
# print(testing_set_vectors)
# testing_predictions = clf.predict_proba(testing_set_vectors)
# print('Predictions: ')
# print(testing_predictions)
# print(len(testing_predictions))
# print(len(ev.get_testing_set_ids()))
#
# output_df = pd.DataFrame()
# output_df['id'] = ev.get_testing_set_ids()
# output_df['toxic'] = testing_predictions[:, 0]
# output_df['severe_toxic'] = testing_predictions[:, 1]
# output_df['obscene'] = testing_predictions[:, 2]
# output_df['threat'] = testing_predictions[:, 3]
# output_df['insult'] = testing_predictions[:, 4]
# output_df['identity_hate'] = testing_predictions[:, 5]
# output_df.to_csv(path_or_buf='predictions.csv')
