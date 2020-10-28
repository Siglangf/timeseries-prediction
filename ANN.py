from sklearn.neural_network import MLPRegressor
import numpy as np


def ANN_train(timeseries, q, hidden_layers, alpha=1e-5, random_state=1):
    '''
    INPUT:
    timeseries: TYPE: DataFrame. Contains column named 'residual'
    q: TYPE: Integer. Decides how many historical values to consider
    hidden_layers: TYPE: Tuple of integers. The number on the i'th position decides number of nodes in hidden layer i

    OUTPUT:
    Returns trained model: TYPE: MLPRegressor
    '''
    X = np.array([timeseries['residual'].iloc[i:i+q]
                  for i in range(len(timeseries['residual'])-q)])
    y = timeseries['residual'].iloc[q:].values
    model = MLPRegressor(solver="lbfgs", alpha=alpha,
                         hidden_layer_sizes=hidden_layers, random_state=random_state)
    model.fit(X, y)
    return model
