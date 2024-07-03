import numpy as np


class LinearRegression:
    """
    A class used to represent a Linear Regression model.

    Attributes
    ----------
    B : numpy.ndarray
        Coefficients of the linear regression model.
    X : numpy.ndarray
        Input feature matrix with intercept term added.
    Y : numpy.ndarray
        Target variable.
    x : pandas.DataFrame
        Input feature matrix before transformation.
    y : pandas.Series
        Target variable before transformation.

    Methods
    -------
    create():
        Computes the coefficients of the linear regression model using the normal equation.
    predict(test):
        Predicts the target variable for given test data.
    """

    def __init__(self, x, y):
        """
        Initializes the LinearRegression model with input features and target variable.

        Parameters
        ----------
        x : pandas.DataFrame
            Input feature matrix.
        y : pandas.Series
            Target variable.
        """
        self.B = None
        self.X = None
        self.Y = None
        self.x = x
        self.y = y

    def create(self):
        """
        Computes the coefficients of the linear regression model using the normal equation.

        The method adds an intercept term to the input feature matrix and calculates the
        coefficients using the formula: B = (X^T * X)^-1 * X^T * Y.
        """
        intercept = np.ones((self.x.shape[0], 1))
        self.X = np.hstack((intercept, self.x.to_numpy()))
        self.Y = self.y.to_numpy()
        self.B = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y

    def predict(self, test):
        """
        Predicts the target variable for given test data.

        Parameters
        ----------
        test : pandas.DataFrame
            Test data for which predictions are to be made.

        Returns
        -------
        numpy.ndarray
            Predicted values for the test data.
        """
        intercept = np.ones((test.shape[0], 1))
        test = np.hstack((intercept, test.to_numpy()))
        return test @ self.B


def sigmoid(z):
    """
    Computes the sigmoid of z.

    Parameters
    ----------
    z : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    """
    A class used to represent a Logistic Regression model.

    Attributes
    ----------
    learning_rate : float
        Learning rate for gradient descent.
    maxiter : int
        Number of iterations for gradient descent.
    weights : numpy.ndarray
        Coefficients of the logistic regression model.
    bias : float
        Intercept term of the logistic regression model.

    Methods
    -------
    fit(x, y):
        Trains the logistic regression model using gradient descent.
    predict_prob(x):
        Predicts the probability of the target variable being 1.
    predict(x, threshold=0.5):
        Predicts the target variable for given test data.
    """

    def __init__(self, learning_rate=0.01, maxiter=1000):
        """
        Initializes the LogisticRegression model with specified learning rate and iterations.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for gradient descent (default is 0.01).
        maxiter : int, optional
            Number of iterations for gradient descent (default is 1000).
        """
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        """
        Trains the logistic regression model using gradient descent.

        Parameters
        ----------
        x : numpy.ndarray
            Input feature matrix.
        y : numpy.ndarray
            Target variable.
        """
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
        """
        Predicts the probability of the target variable being 1.

        Parameters
        ----------
        x : numpy.ndarray
            Input feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted probabilities.
        """
        linear_model = np.dot(x, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, x, threshold=0.5):
        """
        Predicts the target variable for given test data.

        Parameters
        ----------
        x : numpy.ndarray
            Input feature matrix.
        threshold : float, optional
            Threshold for converting predicted probabilities to binary outcomes (default is 0.5).

        Returns
        -------
        numpy.ndarray
            Predicted binary outcomes.
        """
        y_pred_proba = self.predict_prob(x)
        return (y_pred_proba >= threshold).astype(int)
