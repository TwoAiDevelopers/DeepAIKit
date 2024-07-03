import pandas as pd
import numpy as np
from numpy.linalg import inv


class LinearRegression:
    def __init__(self, x, y):
        self.B = None
        self.X = None
        self.Y = None
        self.x = x
        self.y = y

    def create(self):
        intercept = np.ones((self.x.shape[0], 1))
        self.X = np.hstack((intercept, self.x.to_numpy()))
        self.Y = self.y.to_numpy()
        self.B = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y

    def predict(self, test):
        intercept = np.ones((test.shape[0], 1))
        test = np.hstack((intercept, test.to_numpy()))
        return test @ self.B


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, learning_rate=0.01, maxiter=1000):
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.maxiter):
            linear_model = np.dot(x, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_prob(self, x):
        linear_model = np.dot(x, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, x, threshold=0.5):
        y_pred_proba = self.predict_prob(x)
        return (y_pred_proba >= threshold).astype(int)
