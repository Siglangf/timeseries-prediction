# Timeseries Prediction

In this project an extension of Zhang's hybrid forecasting model is implemented. It is extended in the way that it implements feature engineering and multi-step forecasting. 

It is done as a project in Machine Learning course (TDT4173) the fall of 2020 at NTNU (Norwegian University of Science and Technology).

## Dependencies
* Python 3.8
* Pandas
* Scikit-Learn
* Statsmodels
* Numpy
* Datetime
* Matplotlib
* Tqdm

## Structure of Repository

The overall structure of this repository is a bit messy. But once you get a hang of it, it really isn't that much chaos.

There are three main structures: ARIMA, ANN, and the Hybrid. 

The Hybrid consists of both ARIMA and ANN. The usage of ARIMA here is the same as when the ARIMA is running by itself. The ANN however used in the Hybrid is of substantial difference from the ANN doing prediction by itself. To make this difference easily visible, the ANN files that is a part of the prediction by itself has a name starting with **ANN_alone**, while the ones being a part of the Hybrid has names starting with **ANN_in_hybrid**.

There are also some pickle files in the repository. This is where the data 

## Code style

The code that is needed to run the environment is provided in jupyter notebooks. This is done because one can run parts of the code, and store the data between every block, and by that "fool around" with the data without rerunning the entire script all the time :)

## Installation

To be able to run the environment in the way intended, firstly you need to clone the repo into your local computer. Then a recommended step is to open the repo in jupyter notebook via Anaconda (assumed installed). You may ensure that you have the latest versions of the dependencies listed above.

If you encounter any problems with the pickle files, a recommended way is to run the code where they are generated locally, as a regular problem is version handling of these files.

To run either ARIMA, ANN or the Hybrid, make sure to follow the guidelines described below:

### ARIMA


### ANN


### Hybrid




To run the KPI analysis file, follow this guidline:

### KPI analysis
