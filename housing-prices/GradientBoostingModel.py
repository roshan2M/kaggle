import sys
import pandas as pd
import numpy as np

from utils.Preprocessor import Preprocessor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

TRAIN_FILE = './train.csv'
TEST_FILE = './test.csv'

NUM_FEATURES = 20
SK_MODEL_PARAMS = {
    'loss': 'lad',
    'learning_rate': 0.05,
    'max_depth': 9,
    'subsample': 0.7,
    'n_estimators': 500
}
XGB_MODEL_PARAMS = {
    'learning_rate': 0.01,
    'reg_alpha': 5,
    'max_depth': 6,
    'n_jobs': 4,
    'min_child_weight': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'n_estimators': 500
}

SK_MODEL_GRID_PARAMS = {
    'loss': ['lad'],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.7, 0.8],
    'n_estimators': [100, 200, 500, 1000]
}
XGB_MODEL_GRID_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_alpha': [5],
    'max_depth': [6, 9],
    'min_child_weight': [3, 11],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'n_estimators': [100, 200, 500, 1000]
}

def model(trainX, trainY, testData, testX, modelType):
    fitParams = {
        "early_stopping_rounds": 50,
        "eval_metric": "mae",
        "eval_set": [[trainX, trainY]],
        "verbose": False
    }
    if modelType == 'sk':
        model = GradientBoostingRegressor(**SK_MODEL_PARAMS)
    elif modelType == 'xgb':
        model = XGBRegressor(**XGB_MODEL_PARAMS)
    model.fit(trainX, trainY)
    output = {'Id': testData['Id'], 'SalePrice': model.predict(testX)}
    return pd.DataFrame(data=output)

def log_rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))

def grid_search_model(trainX, trainY, modelType):
    if modelType == 'sk':
        model = GradientBoostingRegressor()
        param_grid = SK_MODEL_GRID_PARAMS
    elif modelType == 'xgb':
        model = XGBRegressor()
        param_grid = XGB_MODEL_GRID_PARAMS
    clf = GridSearchCV(
        estimator=model,
        cv=10,
        scoring=make_scorer(log_rmse_loss, greater_is_better=False),
        verbose=1,
        param_grid=param_grid,
        n_jobs=-1)
    clf.fit(trainX, trainY)
    print 'Best Score:', clf.best_score_
    print 'Best Estimator:', clf.best_estimator_
    print 'Best Params:', clf.best_params_

def main():
    modelType = sys.argv[1] # model type can be 'sk' or 'xgb'

    preprocess = Preprocessor()
    train = preprocess.load_data(TRAIN_FILE)
    trainX = np.array(preprocess.fill_missing_values(train.drop(['Id', 'SalePrice'], axis=1)).select_dtypes(exclude=['object']))
    trainY = np.array(preprocess.fill_missing_values(train[['SalePrice']]))

    # modelScore = get_k_fold_cross_validation(trainX, trainY, modelType)
    # modelScores = grid_search_model(trainX, trainY, modelType)

    test = preprocess.load_data(TEST_FILE)
    testX = np.array(preprocess.fill_missing_values(test.drop(['Id'], axis=1)).select_dtypes(exclude=['object']))
    outputDf = model(trainX, trainY, test, testX, modelType)
    outputDf.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()
