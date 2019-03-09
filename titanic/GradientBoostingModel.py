import sys
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Optimized parameters for models
XGB_MODEL_PARAMS = {
    'learning_rate': 0.01,
    'alpha': 5,
    'n_estimators': 200,
    'max_depth': 9,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'scale_pos_weight': 1
}
SK_MODEL_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.7,
    'n_estimators': 100
}

# Apply GridSearchCV to optimize params
XGB_MODEL_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'alpha': [0, 5],
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 6, 9],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'min_child_weight': [3, 6, 9],
    'scale_pos_weight': [1]
}
SK_MODEL_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.7, 0.8],
    'n_estimators': [100, 200, 500, 1000]
}

def read_train_data():
    return pd.read_csv('train.csv')

def read_test_data():
    return pd.read_csv('test.csv')

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

def classify_age_in_bands(df):
    df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].dropna().median())
    df.loc[ df['Age'] <= 10.368, 'AgeBand'] = 0
    df.loc[(df['Age'] > 10.368) & (df['Age'] <= 20.315), 'AgeBand'] = 1
    df.loc[(df['Age'] > 20.315) & (df['Age'] <= 30.263), 'AgeBand'] = 2
    df.loc[(df['Age'] > 30.263) & (df['Age'] <= 40.21), 'AgeBand'] = 3
    df.loc[(df['Age'] > 40.21) & (df['Age'] <= 50.158), 'AgeBand'] = 4
    df.loc[(df['Age'] > 50.158) & (df['Age'] <= 60.105), 'AgeBand'] = 5
    df.loc[(df['Age'] > 60.105) & (df['Age'] <= 70.052), 'AgeBand'] = 6
    df.loc[ df['Age'] > 70.052, 'AgeBand'] = 7
    df.loc[:, 'AgeBand'] = df['AgeBand'].astype(int)
    return df

def clean_titanic_data(df):
    df = df[['Pclass', 'Name', 'Fare', 'Embarked', 'Cabin', 'SibSp', 'Parch', 'Sex', 'Age']]
    df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].dropna().median())
    df.loc[:, 'Embarked'] = df['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
    df.loc[:, 'Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df.loc[:, 'FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df = classify_age_in_bands(df)
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    df.loc[df['FamilySize'] != 1, 'IsAlone'] = 0
    df.loc[:, 'IsAlone'] = df['IsAlone'].astype(int)
    df.loc[:, 'Title'] = df['Name'].apply(get_title).replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare').replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df.loc[:, 'Title'] = df['Title'].map(title_mapping).fillna(0)
    df.drop(['Cabin', 'Name', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)
    return df

def get_kfold_accuracy(model, X, y):
    kfold = StratifiedKFold(n_splits=5, random_state=1)
    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def test_model(X, y, modelType):
    fitParams = {
        "early_stopping_rounds": 30,
        "eval_set": [[X, y]],
        "verbose": True
    }
    if modelType == 'sk':
        model = GradientBoostingClassifier(**SK_MODEL_PARAMS).fit(X, y, **fitParams)
    elif modelType == 'xgb':
        model = XGBClassifier(**XGB_MODEL_PARAMS).fit(X, y, **fitParams)
    get_kfold_accuracy(model, X, y)
    return model

def save_real_model(X, y, modelType):
    model = test_model(X, y, modelType)
    test_data = read_test_data()
    X_test = clean_titanic_data(test_data)
    y_pred = model.predict(X_test)
    save_results_in_csv(y_pred, test_data['PassengerId'])

def grid_search_models(X, y, modelType):
    if modelType == 'sk':
        model = GradientBoostingClassifier()
        clf = GridSearchCV(
            estimator=model,
            cv=5,
            scoring='accuracy',
            verbose=1,
            param_grid=SK_MODEL_PARAM_GRID,
            n_jobs=-1)
    elif modelType == 'xgb':
        model = XGBClassifier()
        clf = GridSearchCV(
            estimator=model,
            cv=5,
            scoring='accuracy',
            verbose=1,
            param_grid=XGB_MODEL_PARAM_GRID,
            n_jobs=-1)
    clf.fit(X, y)
    print 'Best Score:', clf.best_score_
    print 'Best Estimator:', clf.best_estimator_
    print 'Best Params:', clf.best_params_

def save_results_in_csv(predictions, passenger_ids):
    results = pd.DataFrame(index=range(predictions.size))
    results['PassengerId'] = passenger_ids
    results['Survived'] = predictions
    results.to_csv('predictions.csv', index=False)

def main():
    modelType = sys.argv[1]
    input_df = read_train_data()
    X = clean_titanic_data(input_df)
    y = input_df['Survived']
    save_real_model(X, y, modelType)

if __name__ == '__main__':
    main()
