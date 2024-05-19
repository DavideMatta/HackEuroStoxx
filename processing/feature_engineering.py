import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformers for financial indicators
class FinancialIndicators(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        # Drop the Adj. Close
        X = X.reset_index(drop=False).drop('Adj Close', axis=1)
        # Bollinger Bands
        X = pd.concat((X, X.ta.bbands(close=X['Close'], length=20, std=2)), axis=1)
        # Rename the resulting columns without the point to avoid variable name issues with the Pydantic model for the api
        X = X.rename({
                    'BBL_20_2.0': 'BBL_20_2',
                    'BBM_20_2.0': 'BBM_20_2',
                    'BBU_20_2.0': 'BBU_20_2',
                    'BBB_20_2.0': 'BBB_20_2',
                    'BBP_20_2.0': 'BBP_20_2'
                    }, axis=1)
        # RSI
        X['RSI'] = X.ta.rsi(close=X['Close'], length=14)
        return X

# Custom transformer for date handling and converting to numerical values
class DateHandling(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='Date', base_date=None):
        self.date_col = date_col
        self.base_date = base_date if base_date is not None else pd.Timestamp('1970-01-01')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reset index Date as a column if it is not a column yet
        if not(self.date_col in X.columns):
            X = X.reset_index()
        
        # Convert the specified column to datetime format
        X[self.date_col] = pd.to_datetime(X[self.date_col])

        # Extract month and weekday from the datetime column
        X['Month'] = X[self.date_col].dt.month
        X['Weekday'] = X[self.date_col].dt.dayofweek

        # Sine and cosine transformations for month and weekday
        X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
        X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
        X['Weekday_sin'] = np.sin(2 * np.pi * X['Weekday'] / 7)
        X['Weekday_cos'] = np.cos(2 * np.pi * X['Weekday'] / 7)

        # Convert the Date column to a numerical format (days since base_date)
        X['Date_numeric'] = (X[self.date_col] - self.base_date).dt.days

        # Drop the original 'Month', 'Weekday', and 'Date' columns as they have been transformed
        X = X.drop(['Month', 'Weekday'], axis=1)

        return X

# Custom transformer for creating the target column by shifting back the Close price of the next day
class CreateTarget(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Shift the 'Close' column by -1 period
        X['Close_t1'] = X['Close'].shift(-1)
        return X

# Custom transformer for dropping missing values
class DropNA(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna()