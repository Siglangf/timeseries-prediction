
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA


def ARIMA_train(timeseries, p, q, d, prediction_start=20):
    '''
    This function takes a pandas Dataframe with a column 'grid-loss' 
    and return a ARIMA prediction for each timestamp based on p and q as parameters. We need at least 50 data points for training to make a usefull prediction

    Output: Same dataframe, but with columns 'ARIMA_prediction' and 'residual' added
    '''
    grid_loss = timeseries['grid-loss'].values
    train = [grid_loss[i] for i in range(prediction_start)]
    predictions = train.copy()
    residuals = [0 for i in range(prediction_start)]
    for i in range(prediction_start, len(grid_loss)):
        # Initialize model
        model = ARIMA(train, order=(p, d, q))
        # Fit model
        fitted_model = model.fit(disp=0)
        # Create one step ahead forecast based on training
        forecast = fitted_model.forecast()[0][0]
        actual_value = grid_loss[i]
        # Add predictions
        predictions.append(forecast)
        # Add residual
        residuals.append(actual_value-forecast)
        # Add actual value to train for the next forecast
        train.append(forecast)

    timeseries['ARIMA_prediction'] = predictions
    timeseries['residual'] = residuals

    return timeseries
