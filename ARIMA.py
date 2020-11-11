import time
from datetime import timedelta
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from data_extraction import *
import warnings
import numpy as np
warnings.filterwarnings('ignore')
###### Help functions ########


def to_datetime(string):
    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")


def to_string(date):
    return datetime.strftime(date, "%Y-%m-%d %H:%M:%S")


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def ARIMA_train(data, order):
    train = data.copy()
    # Differencing train set
    train['first_difference'] = difference(train, 365*24, 'grid-loss')
    train['second_difference'] = difference(train, 24, 'first_difference')
    train = train.dropna()
    # Construct model
    model = ARIMA(train[['second_difference']], order=order)
    fitted = model.fit(disp=0)

    return fitted


def ARIMA_24_hour_forecast(df, forecast_start, order):
    temp = df.copy()
    # Differencing original data
    temp['year_difference'] = difference(temp, 24*365, 'grid-loss')
    temp['day_difference'] = difference(temp, 24, 'year_difference')
    fitted_model = ARIMA_train(temp[:to_string(forecast_start)], order)

    forecast = fitted_model.forecast(steps=24)[0]
    # Calculate new index
    new_date_index = [to_string(forecast_start + timedelta(hours=i))
                      for i in range(1, 24+1)]

    forecast_df = pd.DataFrame(index=new_date_index)
    forecast_df['ARIMA_differenced'] = forecast

    # Use yearlier values to invert difference
    temp = pd.concat([temp, forecast_df])
    temp['day_inverted'] = invert_difference(
        temp, temp['ARIMA_differenced'], 24, 'year_difference')
    temp['ARIMA'] = invert_difference(
        temp, temp['day_inverted'], 24*365, 'grid-loss')

    # Extracts only the 24 hour forecast
    forecast_df = temp['ARIMA'].dropna()

    return pd.DataFrame(forecast_df)


def difference(df, interval, difference_column):
    temp = df.copy()
    temp['shifted'] = temp[difference_column].shift(interval)
    temp['differenced'] = temp[difference_column]-temp['shifted']
    return temp[['differenced']]


def invert_difference(df, forecast_df, difference_interval, difference_column):
    temp = df.copy()
    temp['shifted'] = temp[difference_column].shift(difference_interval)
    return forecast_df + temp['shifted']


def preprosses_data():
    # choose grid number (current dataset has data from 3 grids so you can choose between 1,2 or 3)
    grid_number = 1
    # Extracting train data
    train = get_timeseries('train', None, 'loss', 1)
    train = pd.DataFrame(train)
    train = train.rename(
        columns={'index': 'timestamp', f'grid{grid_number}-loss': 'grid-loss'})
    # Extracting test data
    test = get_timeseries('test', None, 'loss', 1)
    test = pd.DataFrame(test)
    test = test.rename(
        columns={'index': 'timestamp', f'grid{grid_number}-loss': 'grid-loss'})
    # Merge train and test
    df = pd.concat([train, test])
    return df


def main():
    ######### Hyperparameters ####################
    order = (9, 1, 7)
    ######### DATA EXTRACTION ####################
    data = preprosses_data()

    forecast_df = pd.DataFrame()
    # Required because of differencing of one year and one day and need at least 7 days to make a useful arima prediction
    min_train_size = 365*24 + 24 + 24*7
    # Calculate start date
    start_date = to_datetime(data.index[min_train_size])-timedelta(hours=1)
    # Calculate end date
    end_date = to_datetime(data.index[-1])
    end_date = datetime(end_date.year, end_date.month, end_date.day)
    elapsed_time = 0
    time_df = pd.DataFrame()
    for forecast_date in daterange(start_date, end_date):
        #print(f"Last duration: {elapsed_time}",end='\r')
        print(
            f"STATUS:{to_string(forecast_date)}/{to_string(end_date)} --- Last Duration:{elapsed_time}", end='\r')
        t1 = time.time()
        forecast = ARIMA_24_hour_forecast(
            data[:to_string(forecast_date)], forecast_date, order)
        forecast_df = forecast_df.append(forecast)
        forecast_date += timedelta(days=1)
        # Calculate duration
        t2 = time.time()
        h, rem = divmod(t2-t1, 3600)
        m, s = divmod(rem, 60)
        elapsed_time = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        row = pd.DataFrame([elapsed_time], index=[
                           forecast_date], columns=['Computation time'])
        time_df = time_df.append(row)
        time_df.to_pickle('computation.pkl')

    data = data.join(forecast_df)
    data = data.dropna()
    data['residuals'] = data['grid-loss']-data['ARIMA']
    data.to_pickle('forecast.pkl')


if __name__ == "__main__":
    main()
