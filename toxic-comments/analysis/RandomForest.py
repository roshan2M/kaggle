from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import Evaluation as ev
import DataImport as di

rf_model = RandomForestClassifier(n_estimators=len(di.CATEGORIES))
train_vectors, test_vectors = ev.get_train_test_comment_vectors()
train_classes, test_classes = ev.get_train_test_comment_classes()

for cls in di.CATEGORIES:
    print('CLASS {0}'.format(cls))
    output = train_classes[cls]
    scores = cross_val_score(rf_model, train_vectors, output, scoring='roc_auc')
    print('CV AUC {0}, Average AUC {1}'.format(scores, scores.mean()))
