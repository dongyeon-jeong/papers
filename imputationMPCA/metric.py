import numpy as np

def RMSE(y, yhat):
    error = y - yhat
    rmse = np.sqrt(np.mean(np.square(error)))
    return rmse

def CV(y, yhat):

    error = y - yhat
    mean = np.mean(y)
    cv = np.sqrt(np.sum(np.square(error)) / (len(y) - 1)) / mean * 100
    return cv

def MAPE(y, yhat):
    mape = np.mean(np.abs((y - yhat) / y)) * 100
    return mape

def ForecastSkill(y, yhat, rmse_pers):
    error = y - yhat
    rmse = np.sqrt(np.mean(np.square(error)))
    fs = (1 - np.square(rmse / rmse_pers)) * 100
    return fs
