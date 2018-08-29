from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

import Evaluation as ev
import DataImport as di
import numpy as np


class ExtendedMultiOutputClassifier(MultiOutputClassifier):
    def transform(self, data):
        _output = self.predict_proba(data)
        return np.concatenate(_output, axis=1)


rf_model = RandomForestClassifier(n_estimators=len(di.CATEGORIES), class_weight='balanced')
train_vectors, test_vectors = ev.get_train_test_comment_vectors()
train_classes, test_classes = ev.get_train_test_comment_classes()

for category in di.CATEGORIES:
    print('CLASS {0}'.format(category))
    output = train_classes[category]
    scores = cross_val_score(rf_model, train_vectors, output, scoring='roc_auc')
    print('CV AUC {0}, Average AUC {1}'.format(scores, scores.mean()))

moc = ExtendedMultiOutputClassifier(rf_model)
nnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=0)

clf = Pipeline([('moc_rf', moc), ('nnc', nnc)])
clf.fit(train_vectors, train_classes)

predictions = clf.predict_proba(test_vectors)
diff = predictions - test_classes
msd = np.sum(list(map(lambda x: np.dot(x, x.T), diff.as_matrix()))) / float(len(diff))
print('Mean Squared Error: {0}'.format(msd))
print('Accuracy: {0}'.format(1 - msd))
