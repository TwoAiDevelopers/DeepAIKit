
# Machine Learning Models Documentation

## class LinearRegression
A class used to represent a Linear Regression model.

**Attributes**

- `B` : numpy.ndarray  
  Coefficients of the linear regression model.
- `X` : numpy.ndarray  
  Input feature matrix with intercept term added.
- `Y` : numpy.ndarray  
  Target variable.
- `x` : pandas.DataFrame  
  Input feature matrix before transformation.
- `y` : pandas.Series  
  Target variable before transformation.

**Methods**

- `create()`:  
  Computes the coefficients of the linear regression model using the normal equation.
- `predict(test)`:  
  Predicts the target variable for given test data.

### __init__
Initializes the LinearRegression model with input features and target variable.

**Parameters**

- `x` : pandas.DataFrame  
  Input feature matrix.
- `y` : pandas.Series  
  Target variable.

### create
Computes the coefficients of the linear regression model using the normal equation.

The method adds an intercept term to the input feature matrix and calculates the coefficients using the formula: `B = (X^T * X)^-1 * X^T * Y`.

### predict
Predicts the target variable for given test data.

**Parameters**

- `test` : pandas.DataFrame  
  Test data for which predictions are to be made.

**Returns**

- `numpy.ndarray`  
  Predicted values for the test data.

## def sigmoid(z)
Computes the sigmoid of z.

**Parameters**

- `z` : numpy.ndarray  
  Input array.

**Returns**

- `numpy.ndarray`  
  Sigmoid of the input array.

## class LogisticRegression
A class used to represent a Logistic Regression model.

**Attributes**

- `learning_rate` : float  
  Learning rate for gradient descent.
- `maxiter` : int  
  Number of iterations for gradient descent.
- `weights` : numpy.ndarray  
  Coefficients of the logistic regression model.
- `bias` : float  
  Intercept term of the logistic regression model.

**Methods**

- `fit(x, y)`:  
  Trains the logistic regression model using gradient descent.
- `predict_prob(x)`:  
  Predicts the probability of the target variable being 1.
- `predict(x, threshold=0.5)`:  
  Predicts the target variable for given test data.

### __init__
Initializes the LogisticRegression model with specified learning rate and iterations.

**Parameters**

- `learning_rate` : float, optional  
  Learning rate for gradient descent (default is 0.01).
- `maxiter` : int, optional  
  Number of iterations for gradient descent (default is 1000).

### fit
Trains the logistic regression model using gradient descent.

**Parameters**

- `x` : numpy.ndarray  
  Input feature matrix.
- `y` : numpy.ndarray  
  Target variable.

### predict_prob
Predicts the probability of the target variable being 1.

**Parameters**

- `x` : numpy.ndarray  
  Input feature matrix.

**Returns**

- `numpy.ndarray`  
  Predicted probabilities.

### predict
Predicts the target variable for given test data.

**Parameters**

- `x` : numpy.ndarray  
  Input feature matrix.
- `threshold` : float, optional  
  Threshold for converting predicted probabilities to binary outcomes (default is 0.5).

**Returns**

- `numpy.ndarray`  
  Predicted binary outcomes.
