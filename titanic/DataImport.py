import pandas as pd

train_file = "train.csv"
test_file = "test.csv"

titanic_data = pd.read_csv(train_file)
X_train = titanic_data.drop('Survived', axis=1)
y_train = titanic_data['Survived']
X_test = pd.read_csv(test_file)
