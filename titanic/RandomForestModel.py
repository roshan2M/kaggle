import pandas as pd
import DataImport as di

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

COLUMN_NAMES = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age_category_Missing',
                'Age_category_Infant', 'Age_category_Child', 'Age_category_Teenager',
                'Age_category_Young Adult', 'Age_category_Adult', 'Age_category_Senior',
                'Sex_female', 'Sex_male']

rf = RandomForestClassifier()

titanic_train_data = di.get_clean_train_data()
train_X = titanic_train_data[COLUMN_NAMES]
train_y = titanic_train_data['Survived']

titanic_test_data = di.get_clean_test_data()
test_X = titanic_test_data[COLUMN_NAMES]
test_y = di.get_titanic_test_results()['Survived']

rf.fit(train_X, train_y)
predictions = rf.predict(test_X)

results = pd.DataFrame(index=range(predictions.size), columns=[])
results["PassengerId"] = titanic_test_data["PassengerId"]
results["Survived"] = predictions
results.to_csv("Titanic Predictions Random Forest.csv", index=False)

score = accuracy_score(test_y, predictions)
print('Score: ' + str(score))
