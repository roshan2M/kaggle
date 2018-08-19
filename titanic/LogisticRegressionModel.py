import DataImport as di

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

AGE_CUTS = [-1, 0, 5, 12, 18, 35, 60, 100]
AGE_LABELS = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']
COLUMN_NAMES = ['Pclass_1', 'Pclass_2', 'Pclass_3',
                'Age_category_Infant', 'Age_category_Child', 'Age_category_Teenager',
                'Age_category_Young Adult', 'Age_category_Adult', 'Age_category_Senior',
                'Sex_female', 'Sex_male']


def build_train_test_sets() -> list:
    titanic_train_data = di.get_clean_train_data(AGE_CUTS, AGE_LABELS)
    all_X = titanic_train_data[COLUMN_NAMES]
    all_y = titanic_train_data['Survived']
    return train_test_split(all_X, all_y, test_size=0.2, random_state=0)


def get_lr_model_accuracy() -> float:
    train_X, test_X, train_y, test_y = build_train_test_sets()
    lr_model = LogisticRegression()
    lr_model.fit(train_X, train_y)
    predictions = lr_model.predict(test_X)
    return accuracy_score(test_y, predictions)


print(get_lr_model_accuracy())  # 0.810056
