import numpy as np


def rmse(y, y_pred): # Calculate Root Mean Squared Error (RMSE)

    return np.sqrt(np.mean((y - y_pred) ** 2))


def mae(y, y_pred):  # Calculate Mean Absolute Error (MAE)

    return np.mean(np.abs(y - y_pred))


def mse(y, y_pred):  # Calculate Mean Squared Error (MSE)

    return np.mean((y - y_pred) ** 2)


def calculate_r2(y, y_pred):  # Calculate the R-Squared (Coeff of determination)
    ss_res = np.sum((np.array(y) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y) - np.mean(np.array(y))) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_mape(y, y_pred):  # Calculate the Mean Absolute Percentage Error (MAPE).

    return np.mean(np.abs((np.array(y) - np.array(y_pred)) / np.array(y))) * 100


def calculate_mbd(y, y_pred):  # Calculate the Mean Bias Deviation (MBD).

    return np.mean(np.array(y_pred) - np.array(y))


def calculate_explained_variance(y, y_pred):  # Calculate the Explained Variance Score.

    return 1 - np.var(np.array(y) - np.array(y_pred)) / np.var(np.array(y))


def calculate_medae(y, y_pred):  # Calculate the Median Absolute Error (MedAE).

    return np.median(np.abs(np.array(y) - np.array(y_pred)))
