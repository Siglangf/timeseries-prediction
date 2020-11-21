


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import data_extraction


past_days = 40
nRowsRead = None
grid_number = 1



def data_ANN(past_days):
    # input values
    # past_days is the number of past days of the grid_loss that the ANN will use
    loss_train_series = data_extraction.get_timeseries("train", nRowsRead, 'loss', grid_number)
    loss_test_series = data_extraction.get_timeseries("test", nRowsRead, 'loss', grid_number)
    load_train_series = data_extraction.get_timeseries("train", nRowsRead, 'load', grid_number)
    load_test_series = data_extraction.get_timeseries("test", nRowsRead, 'load', grid_number)
    
    loss_series = loss_train_series.append(loss_test_series)
    load_series = load_train_series.append(load_test_series)
    
    X = []
    y = []
    
    
    for i in range(len(loss_series) - past_days*24):
        X.append(loss_series[i:i+past_days*24:24].values + load_series[i:i+past_days*24:24].values)
        y.append(loss_series[i + past_days*24])
    
    return X, y


def getgsCV(numberOfPastVals):
    X, y = data_ANN(numberOfPastVals)

    train_size = len(data_extraction.get_timeseries("train", nRowsRead, 'loss', grid_number)) - past_days*24

    X_train = X[:train_size]
    y_train = y[:train_size]

    
    model = MLPRegressor(solver="adam", 
                         alpha=1e-5, 
                         random_state=1, 
                         max_iter=20000,
                         tol=1e-4)
    parameters = {'solver': ['adam'], 'hidden_layer_sizes' : [(i) for i in range(1,numberOfPastVals*2,1)] + [(i,i) for i in range(1,round(numberOfPastVals*2/2),1)] + [(i,i,i) for i in range(1,round(numberOfPastVals*2/3),1)]}

    gsCV = GridSearchCV(model , parameters, verbose=10)
    gsCV.fit(X_train, y_train)
    return gsCV


gsCVList = []
past_vals = [i for i in range(1,70,5)]



for val in past_vals:
    gsCVList.append(getgsCV(val))


CVdf = pd.DataFrame(index = past_vals)

bestScores = []
bestHiddenLayers = []

for i in range(len(gsCVList)):
    bestScores.append(gsCVList[i].best_score_)
    bestHiddenLayers.append(gsCVList[i].best_params_['hidden_layer_sizes'])


CVdf['past'] = past_vals
CVdf['scores'] = bestScores
CVdf['hidden layers'] = bestHiddenLayers


CVdf.to_pickle("hyper_parameters_ANN.pkl")

