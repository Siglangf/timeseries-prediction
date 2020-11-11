from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import data_extraction
import ARIMA
from datetime import datetime
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def ANN_train(X_train, y_train, hidden_layers, alpha=1e-1, random_state=1):
    '''
    INPUT:
    timeseries: TYPE: DataFrame. Contains column named 'residual'
    q: TYPE: Integer. Decides how many historical values to consider
    hidden_layers: TYPE: Tuple of integers. The number on the i'th position decides number of nodes in hidden layer i

    OUTPUT:
    Returns trained model: TYPE: MLPRegressor
    '''
    model = MLPRegressor(solver="lbfgs",
                         alpha=alpha,
                         hidden_layer_sizes=hidden_layers,
                         random_state=random_state,
                         max_iter=20000,
                         tol=1e-4)  # default tolerance
    model.fit(X_train, y_train)
    return model


def to_datetime(string):
    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")


def to_string(date):
    return datetime.strftime(date, "%Y-%m-%d %H:%M:%S")


def data_ANN_train(past_vals, df):
    # input values
    # past_vals is the number of past values of the residuals that the ANN will use
    residuals = df["residuals"]

    load_series_train = data_extraction.get_timeseries(
        "train", None, 'load', 1)
    load_series_test = data_extraction.get_timeseries("test", None, 'load', 1)

    X = []
    y = []

    for i in range(len(residuals.values) - len(load_series_test) - past_vals):
        X_el = residuals[i:i+past_vals]
        t = datetime.strptime(
            residuals.index[i+past_vals], "%Y-%m-%d %H:%M:%S")

        X_el_additional = []
        X_el_additional.append(t.hour)
        X_el_additional.append(t.month)

        # add the load of the predicted value
        X_el_additional.append(load_series_train[i+past_vals])

        X_el_additional = pd.Series(X_el_additional, name=X_el.name)

        df = pd.DataFrame(X_el.append(X_el_additional, ignore_index=True))

        nd = df.values

        nd_new = np.ndarray(shape=len(df.values))

        for j in range(len(nd)):
            nd_new[j] = nd[j, 0]
        X.append(nd_new)
        y.append(residuals.iloc[i+past_vals])

    return X, y


def data_ANN_test(past_vals, df):
    residuals = df["residuals"]
    load_series_test = data_extraction.get_timeseries("test", None, 'load', 1)

    X = []

    for i in range(len(load_series_test)):
        if i % 24 == 0:
            X_el = residuals[len(load_series_test)+i -
                             past_vals:len(load_series_test)+i]
            # print(len(X_el))

        # if i == 0:
        #    print(X_el)

        t = datetime.strptime(load_series_test.index[i], "%Y-%m-%d %H:%M:%S")

        X_el_additional = []
        X_el_additional.append(t.hour)
        X_el_additional.append(t.month)

        # add the load of the predicted value
        X_el_additional.append(load_series_test[i])

        X_el_additional = pd.Series(X_el_additional, name=X_el.name)

        df = pd.DataFrame(X_el.append(X_el_additional, ignore_index=True))
        nd = df.values

        nd_new = np.ndarray(shape=len(df.values))

        for j in range(len(nd)):
            nd_new[j] = nd[j, 0]

        X.append(nd_new)
        X_el = X_el[1:]
    return X


def ANN_pred(model, X_test):
    X = X_test.copy()
    y_test = []
    for i in range(len(X)):
        X_el = X[i]
        X_losses = X_el[:-3]

        for j in range(i % 24):
            y_test_el = np.ndarray(shape=1)
            v = y_test[- (i % 24 - j)]
            y_test_el[0] = v
            X_losses = np.append(X_losses, y_test_el)

        X[i] = np.append(X_losses, X_el[-3:])

        val = model.predict([X[i]])

        y_test.append(val[0])
    return y_test


def main():

    #########Parameters##############
    past_vals = 100  # Number of past values to include in input layer
    nRowsRead = None  # Hard coded
    grid_number = 1
    hidden_layer = (4)
    # Read df with arima forecast
    df = pd.read_pickle('forecast.pkl')
    # Dividing
    logging.info("Preprocessing training data")
    X, y = data_ANN_train(past_vals, df)
    # Training model
    logging.info("Training...")
    model = ANN_train(X, y, hidden_layer)
    logging.info("Preprocessing test data")
    X_test = data_ANN_test(past_vals, df)
    logging.info("Predicting...")
    predicted_ARIMA_error = ANN_pred(model, X_test)

    df['pred_error'] = np.nan
    df['pred_error'].iloc[-len(predicted_ARIMA_error):] = predicted_ARIMA_error
    # Adds predicted error with ARIMA forcast to get Hybrid forecast
    df['Hybrid'] = df['ARIMA'] + df['pred_error']
    df = df.dropna()  # Removing train set

    pd.to_pickle(df[['grid-loss', 'Hybrid', 'ARIMA']], 'Hybrid.pkl')
    logging.info("Finished.")


if __name__ == "__main__":
    main()
