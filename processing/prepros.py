import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

def get_X_y(df):
    X = df.drop('Close', axis=1)
    y = df['Close']


#def three_dim(X, y):

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


    def transform(self, X):
        X, y = get_X_y(X)
        out = three_dim(X, y)
        return out
    


    window_size = 5

# Initialize an empty list to store 3D arrays
X_3d_list = []

# Iterate through each window
for i in range(len(X) - window_size + 1):
    # Extract the data for the current window
    window_data = X.iloc[i:i+window_size]
    # Convert the window data to a 2D numpy array
    window_array = window_data.drop(columns=['Date']).to_numpy()
    # Append the 2D array with the date column included as the first feature
    window_array_with_time = np.insert(window_array, 0, window_data['Date'], axis=1)
    # Append the 2D array to the list
    X_3d_list.append(window_array_with_time)

# Stack the list of 3D arrays into a single 3D numpy array
X_3d = np.stack(X_3d_list)