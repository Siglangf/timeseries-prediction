
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import data_extraction


nRowsRead = None
grid_number = 1
df = pd.read_pickle('forecast1.pkl')


def data_ANN(past_weeks, df):
    # input values
    # past_vals is the number of past values of the residuals that the ANN will use
    residuals = df["residuals"]

    X = []
    y = []
    
    for i in range(len(residuals.values) - past_weeks*24*7):
        X_el = residuals[i:i+24*7*past_weeks:24*7]
        X.append(X_el.values)
        y.append(residuals.iloc[i+past_weeks*24*7])
    
    return X, y




def getgsCV(numberOfPastVals):
    X, y = data_ANN(numberOfPastVals, df)
    
    train_size_percentage = 0.80

    train_size = int(len(X)*train_size_percentage)

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
past_vals = [i for i in range(1,30,1)]



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


CVdf.to_pickle("hyper_parameters_hybrid_short.pkl")


