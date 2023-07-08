import os
import numpy as np
import copy


def make_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

def RMSE(real, pred):

    error = real - pred
    return np.sqrt(np.nanmean(np.square(error)))

def CV(y, yhat):

    error = y - yhat
    mean = np.mean(y)
    cv = np.sqrt(np.sum(np.square(error)) / (y.size - 1)) / mean * 100
    return cv

def MAPE(y, yhat):
    mape = np.mean(np.abs((y - yhat) / y)) * 100
    return mape

def ForecastSkill(y, yhat, rmse_pers):
    rmse = RMSE(y, yhat)
    # print(rmse, rmse_pers)
    fs = (1 - np.square(rmse / rmse_pers)) * 100
    return fs
