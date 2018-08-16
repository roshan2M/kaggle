from sklearn.linear_model import LogisticRegression

from DataImport import *

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
y_prediction = logistic_regression_model.predict(X_test)
print(logistic_regression_model.score(X_train, y_train))
