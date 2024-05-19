import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def _init_(self, X, y):
        self.X = X
        self.y = y

    def _len_(self):
        return len(self.X)

    def _getitem_(self, index):
        return self.X[index], self.y[index]

class Preprocessing(BaseEstimator):
    def __init__(self, window_size=5, batch_size=32):
        self.window_size = window_size
        self.batch_size = batch_size

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
                X,
                y,
                train_size: float = 0.7,
                valid_size: float = 0.2,
                forecast_size: float = 0.02):
        '''
        Splits the dataset into training, validation, testing, and forecasting subsets.

        This method divides the input dataset into four distinct subsets: training, validation, testing, and forecasting.
        The division ratios are determined by the `train_size`, `valid_size`, and `forecast_size` parameters.

        Parameters:
        - X (numpy.ndarray): The input features dataset.
        - y (numpy.ndarray): The input labels dataset.
        - train_size (float): The proportion of the dataset to include in the training split. Default is 0.7.
        - valid_size (float): The proportion of the dataset to include in the validation split. Default is 0.2.
        - forecast_size (float): The proportion of the dataset to include in the forecasting split. Default is 0.02.

        Returns:
        - tuple: A tuple containing eight numpy arrays representing the split datasets:
            - X_train, y_train: The training features and labels datasets.
            - X_val, y_val: The validation features and labels datasets.
            - X_test, y_test: The testing features and labels datasets.
            - X_forecast, y_forecast: The forecasting features and labels datasets.
        '''
        X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=(1-train_size), shuffle=False)
        X_val, X_test_remainder, y_val, y_test_remainder = train_test_split(X_remainder, y_remainder, test_size=(1 - valid_size/(1-train_size)), shuffle=False)
        X_test, X_forecast, y_test, y_forecast = train_test_split(X_test_remainder, y_test_remainder, test_size=(1 - forecast_size / (1 - valid_size/(1-train_size)) / (1-train_size)), shuffle=False)
        
        assert (1-train_size) == 0.3
        assert (1 - valid_size/(1-train_size)) == 1/3
        assert (1 - forecast_size / (1 - valid_size/(1-train_size)) / (1-train_size)) == 0.2

        return X_train, y_train, X_val, y_val, X_test, y_test, X_forecast, y_forecast
    
    def agg_tv(self, X_train, y_train, X_val, y_val):
        """
        Aggregate Training and Validation Data

        This method concatenates the training and validation datasets along the specified axis,
        effectively creating a combined dataset for further processing or analysis.

        Parameters:
        - X_train (numpy.ndarray): The training features dataset.
        - y_train (numpy.ndarray): The training labels dataset.
        - X_val (numpy.ndarray): The validation features dataset.
        - y_val (numpy.ndarray): The validation labels dataset.

        Returns:
        - tuple: A tuple containing two concatenated numpy arrays:
            - X_tv (numpy.ndarray): The combined features dataset from both training and validation sets.
            - y_tv (numpy.ndarray): The combined labels dataset from both training and validation sets.
        """
        X_tv = np.concatenate((X_train, X_val), axis=0)
        y_tv = np.concatenate((y_train, y_val), axis=0)
        return X_tv, y_tv
        
    def df_to_tdl(self, X, y):
        dataset = CustomDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def transform(self, X):
        X, y = self.get_X_y(X)
        X_3d, y = self.three_dim(X, y)
        return X_3d, y