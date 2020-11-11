import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cleanData(df):
    # remove data that is labeled incorrect
    df = df[df['has incorrect data'] == False]
    df = df.select_dtypes(include=[np.number])  # keep only numerical columns
    return df


# %%
def getFeature(df, number, feature):
    df = df[f'grid{number}-{feature}']
    df = df.dropna()  # if there is any NaN data in the feature, remove it (in the loss feature there isn't but this is safety for future)
    return df


def plotFeature(fd, wf, fu):
    fd.plot(figsize=(30, 10), label=f'{wf}', linewidth=1)
    plt.title(f'Grid {wf} historic data', fontsize=20)
    plt.xlabel('Date and time', fontsize=18)
    plt.ylabel(f'Grid {wf} ({fu})', fontsize=18)
    plt.xticks(fontsize=14)
    plt.legend()


def get_timeseries(name_of_data, n_rows_read, wanted_feature, grid_number):
    data = pd.read_csv(f'./DATASET/data/{name_of_data}.csv',
                       index_col=0, nrows=n_rows_read)
    #data = cleanData(data)
    feature_data = getFeature(data, grid_number, wanted_feature)

    return feature_data
