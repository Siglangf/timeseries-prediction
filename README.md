# Timeseries Prediction

In this project an extension of Zhang's hybrid forecasting model is implemented. It is extended in the way that it implements feature engineering and multi-step forecasting. 

It is done as a project in Machine Learning course (TDT4173) the fall of 2020 at NTNU (Norwegian University of Science and Technology).

## Dependencies
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

The Hybrid consists of both ARIMA and ANN. The usage of ARIMA here is the same as when the ARIMA is running by itself. The ANN however used in the Hybrid is of substantial difference from the ANN doing prediction by itself. To make this difference easily visible, the ANN files that is a part of the prediction by itself has a name starting with ANN_alone, while the ones being a part of the Hybrid has names starting with ANN_in_hybrid.