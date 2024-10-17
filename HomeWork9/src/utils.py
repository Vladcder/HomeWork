import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb



def delete_outliers(train):
    Q1 = train['price'].quantile(0.25)
    Q3 = train['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    upper_array = np.where(train['price'] >= upper)[0]
    lower_array = np.where(train['price'] <= lower)[0]
    train.drop(index=upper_array, inplace=True)
    train.drop(index=lower_array, inplace=True)
    return  train

def prepare_data():
    train = pd.read_csv("data/train.csv", encoding="utf-8")
    train = train.drop(
        columns=['product_name', 'period', 'postcode', 'address_name', 'city', 'settlement', 'object_type', 'district', 'area',
                 'description', 'source'])
    train=delete_outliers(train)
    train = train.dropna()
    train['lat']=round(train['lat'], 2)
    train['lon'] = round(train['lon'], 2)
    train['rooms'] = train['rooms'].astype('int8')
    train['floor'] = train['floor'].astype('int16')
    #train = pd.get_dummies(train, columns=['floor'], drop_first=True)
    return train



def train_model(train):
    X_train, X_test, y_train, y_test = train_test_split(
        train.drop(columns='price'),
        train['price'],
        random_state=20,
        test_size=0.2,
        shuffle=True
    )
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 9,
        'learning_rate': 0.09,
        'n_estimators': 521,
        'subsample': 0.88,
        'colsample_bytree': 1,
        'random_state': 20,
        'gamma':1,
        'alpha':0.137,
        'lambda':0.97,
        'min_child_weight':0.2

    }
    model = Pipeline([('scaler',StandardScaler()),('regressor', xgb.XGBRegressor(**params))])
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    print(f'RMSE: {mean_squared_error(y_test, y_preds, squared=False)}')


    with open('lr_fitted.pkl', 'wb') as file:
        pickle.dump(model, file)



def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model