# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:28:58 2022

@author: wcfda
"""
import pickle
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


# arrange data to the form that could be used by knn algorithm
# each row/observation is a consecutive time series covering 'X_time_range' + 'y_time_range' hours
# we want to predict 'y_time_range' part time series based on 'X_time_range' part
# there is a time lag of 10 min between every row and its previous row
# suppose a row records a time series ranging from time t to t + (X_time_range + y_time_range)*hours, 
# the time series in the previous row will range from t - 10 min to t + (X_time_range + y_time_range)*hours - 10 min
# the time lag between the last and the first row is called row_time_range, which = number of row * 10 min

def ws_data_arrangement(ws_data, row_time_range, X_time_range, y_time_range):
    n_row = row_time_range * 6
    n_col_X = X_time_range * 6
    n_col_y = y_time_range * 6
    data = []
    for i in range(n_row):
        # the +1 is y: wind speed to predict
        row_data = ws_data[i:i + n_col_X + n_col_y].tolist()
        data.append(row_data)
    data = pd.DataFrame(data)
    
    colname = []
    for i in range(n_col_X):
        colname.append('ws_x{}'.format(i))
    for i in range(n_col_y):
        colname.append('ws_y{}'.format(i))
    data.columns = colname
    return data


def knn_train(X_train, Y_train, params = None):
    if params is None:
        parameters = {"n_neighbors": range(1, 10), "weights": ["uniform", "distance"],}
        model = GridSearchCV(KNeighborsRegressor(), parameters)
    else:
        model = KNeighborsRegressor(n_neighbors = params[0], weights = params[1])
    
    model.fit(X_train, Y_train)
    
    return model


# treat X_time_range part as predictors to predict the first element of y_time_range time series,
# consider the predicted value as ground turth and predict the second element 
# based on X_time_range + the first element predicted. Keep adding the predicted elements as 
# predictors/exploratory variables to predict the rest of y_time_range time series
def iterative_knn(X_train, X_test, Y_train, Y_test, params = None):
    test_pred = []
    for colname in Y_train.columns:
        y_train = Y_train[colname]
        model = knn_train(X_train, y_train, [4, 'distance'])
        pred = list(model.predict(X_test))
        test_pred.append(pred)
        X_train[colname]= y_train
        X_test[colname] = pred
        
    test_pred = np.transpose(np.array(test_pred))
    
    return test_pred


def eva(Y_pred, Y_true):
    assert type(Y_pred) == np.ndarray and type(Y_true) == np.ndarray, 'input should be array-like'
    rmse = sum(np.mean((Y_true - Y_pred)**2, axis = 1))
    mae = sum(np.mean(abs(Y_true - Y_pred), axis = 1))
    
    return rmse, mae


def ws_prediction(data, filename):
        colname_X = [colname for colname in data.columns if 'x' in colname]
        colname_Y = [colname for colname in data.columns if 'y' in colname]
        X = data[colname_X]
        Y = data[colname_Y]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)
        score = {}

        start_time = time()
        knn = knn_train(X_train, Y_train)
        test_pred = knn.predict(X_test)
        end_time = time()
        rmse, mae = eva(test_pred, np.array(Y_test))
        score['knn'] = [rmse, mae, end_time - start_time]
  
        
        start_time = time()
        test_pred = iterative_knn(X_train, X_test, Y_train, Y_test, [4, 'distance'])
        end_time = time()
        rmse, mae = eva(test_pred, np.array(Y_test))
        score['knn_iterative'] = [rmse, mae, end_time - start_time]
        
        with open(data_path + filename, 'wb') as fp:
            pickle.dump(score, fp)




if __name__ == '__main__':
    data_path = r'...' # the directory your data in
    filename = 'clean_data.csv'
    data_full = pd.read_csv(data_path + filename)
    
    turbID = 1
    data_one_turbine = data_full[data_full['TurbID'] == turbID]
    data_short = ws_data_arrangement(data_one_turbine['Wspd'], 7*24, 15*24, 2*24)
    data_long = ws_data_arrangement(data_one_turbine['Wspd'], 7*24, 60*24, 2*24)
            
    ws_prediction(data_short, 'knn_eva_short')
    ws_prediction(data_long, 'knn_eva_long')
    with open(data_path + 'score_short', 'rb') as fp:
            knn_info = pickle.load(fp)

