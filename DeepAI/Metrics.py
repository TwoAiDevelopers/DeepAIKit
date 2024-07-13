import numpy as np


def rmse(y, y_pred):  # Calculate Root Mean Squared Error (RMSE)

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


def confusion_matrix(y_true, y_pred):  # Calculate the confusion matrix.
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN


def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def precision(y_pred, y_true):
    TP, _, FP, _ = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) != 0 else 0


def recall(y_true, y_pred):
    TP, _, _, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) != 0 else 0


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0


def specificity(y_true, y_pred):
    _, TN, FP, _ = confusion_matrix(y_true, y_pred)
    return TN / (TN + FP) if (TN + FP) != 0 else 0


def negative_predictive_value(y_true, y_pred):
    _, TN, _, FN = confusion_matrix(y_true, y_pred)
    return TN / (TN + FN) if (TN + FN) != 0 else 0
