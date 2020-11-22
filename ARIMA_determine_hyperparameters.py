from statsmodels.tsa.arima_model import ARIMA
from data_extraction import *
import warnings
import logging
logging.basicConfig(filename='execution.log', level=logging.INFO)
logging.info("EXECUTION STARTED")

warnings.filterwarnings('ignore')

# choose the appropriate data (train, test or test_backfilled_missing_data)
name_of_data = 'train'
n_rows_read = None  # choose number of rows (use None to display all rows)
# choose grid number (current dataset has data from 3 grids so you can choose between 1,2 or 3)
grid_number = 1
# choose name of feature (could be load, loss, temp, or predictive values)
wanted_feature = 'loss'
feature_unit = 'MWh'  # change to Kelvin if the feature is temp (temperature)
df = get_timeseries(name_of_data, n_rows_read, wanted_feature, 1)

df = pd.DataFrame(df)
df = df.rename(columns={'index': 'timestamp',
                        f"grid{grid_number}-loss": 'grid-loss'})


def difference(df, interval):
    temp = df.copy()
    temp['shifted'] = temp['grid-loss'].shift(interval)
    return pd.DataFrame(temp[['shifted', 'grid-loss']].diff(axis=1)['grid-loss'])


def evaluate_ARIMA_model(df, order, difference_interval=365*24, steps=24):
    # Dividing test and train set
    test = df[len(df)-steps:]
    train = df[:len(df)-steps]
    # Differencing train set

    # REMARK:dropping the values from 0-differnce_interval
    differenced = difference(train, difference_interval).dropna()

    # Construct model
    model = ARIMA(differenced, order=order)
    fitted = model.fit(disp=0)
    # Get prediction
    differenced_forecast = fitted.forecast(steps=steps)[0]
    test['forecast'] = differenced_forecast
    # Invert difference
    test['forecast'] += train['grid-loss'][-difference_interval:-
                                           difference_interval+steps].values

    # Calculate mean squared error
    diff_squared = test.diff(axis=1)**2
    return diff_squared.mean()['forecast']


def grid_search_ARIMA(df, p_values, d_values, q_values):

    # Search for best parameters for the ARIMA model
    best_order = ()
    best_score = float("inf")
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    logging.info(
                        f"Evaluating ARIMA model of order {p},{d},{q}")
                    score = evaluate_ARIMA_model(df, (p, d, q))
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                    logging.info(f"Order=({p},{d},{q}), MSE={score}")
                except:
                    logging.info(f"Order=({p},{d},{q}) FAILED")
                    continue

    return best_score, best_order


plist = [i for i in range(5, 15)]
dlist = [1]
qlist = [i for i in range(5, 15)]

logging.info(f"Best score: {grid_search_ARIMA(df, plist, dlist, qlist)}")
