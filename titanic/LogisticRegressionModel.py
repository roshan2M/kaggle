import DataImport as di
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLUMN_NAMES = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age_category_Missing',
                'Age_category_Infant', 'Age_category_Child', 'Age_category_Teenager',
                'Age_category_Young Adult', 'Age_category_Adult', 'Age_category_Senior',
                'Sex_female', 'Sex_male']


def build_train_set() -> list:
    titanic_train_data = di.get_clean_train_data()
    all_X = titanic_train_data[COLUMN_NAMES]
    all_y = titanic_train_data['Survived']
    return [all_X, all_y]


def get_lr_model() -> LogisticRegression:
    all_x, all_y = build_train_set()
    lr_model = LogisticRegression()
    lr_model.fit(all_x, all_y)
    return lr_model


def save_results_in_csv(predictions: list, test_set: pd.DataFrame):
    results = pd.DataFrame(index=range(predictions.size), columns=[])
    results["PassengerId"] = test_set["PassengerId"]
    results["Survived"] = predictions
    results.to_csv("Titanic Predictions.csv", index=False)


def get_lr_model_test_accuracy() -> float:
    titanic_test_set = di.get_clean_test_data()
    titanic_test_X = titanic_test_set[COLUMN_NAMES]
    titanic_test_y = di.get_titanic_test_results()['Survived']
    lr_model = get_lr_model()
    predictions = lr_model.predict(titanic_test_X)
    save_results_in_csv(predictions, titanic_test_set)
    return accuracy_score(titanic_test_y, predictions)


print(get_lr_model_test_accuracy())
