import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

class Preprocessing(BaseEstimator):
    def __init__(self):
        self.window_size = 5

    def fit(self, X, y=None):
        return self
    
    def get_X_y(self, X):
        y = X['Close']
        X = X.drop('Close', axis=1)
        return X, y
    
    def three_dim(self, X, y):
        y = y[4:].reset_index(drop=True).values.reshape(-1, 1, 1)

        X_3d_list = []
        for i in range(len(X) - self.window_size + 1):
            # Extract the data for the current window
            window_data = X.iloc[i:i+self.window_size]
            # Convert the window data to a 2D numpy array
            window_array = window_data.drop(columns=['Date']).to_numpy()
            # Append the 2D array with the date column included as the first feature
            window_array_with_time = np.insert(window_array, 0, window_data['Date'], axis=1)
            # Append the 2D array to the list
            X_3d_list.append(window_array_with_time)
        # Stack the list of 3D arrays into a single 3D numpy array
        X_3d = np.stack(X_3d_list)

        return X_3d, y
    
    def tvtf_split(self,
                   X_3d,
                   y,
                   train_size: float = 0.7,
                   valid_size: float = 0.2,
                   test_size: float = 0.08,
                   forecast_size: float = 0.02):
        '''
        Train, test, valid, forecast splitter
        '''
        X_train, X_remainder, y_train, y_remainder = train_test_split(X_3d, y, test_size=(1-train_size), shuffle=False)
        X_val, X_test_remainder, y_val, y_test_remainder = train_test_split(X_remainder, y_remainder, test_size=0.333, shuffle=False)
        X_test, X_forecast, y_test, y_forecast = train_test_split(X_test_remainder, y_test_remainder, test_size=0.2, shuffle=False)

    def transform(self, X):
        X, y = self.get_X_y(X)
        X_3d, y = self.three_dim(X, y)
        return X_3d, y