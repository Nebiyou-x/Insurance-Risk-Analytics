import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

class Modelling:
    def __init__(self, data):
        self.data = data


    def preprocess(self, num_cols: list)->pd.DataFrame:
        '''
        This funciton normalizes teh data using standard scaler

        Parameters:
            num_cols(list): list of numerical columsn to process

        Returns:
            pd.DataFrame
        '''
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[num_cols])
        scaled_data.DataFrame(scaled_data, columns=num_cols)

        return scaled_data


    def split_data(self, X, y, test_size = 0.2):
        '''
        Splits the data to training and testing

        Parameters:
            X: Features
            y: label

        Returns: 
            X_train, X_test, y_train, y_test 
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test 


    def model_testing(self, model, X_test, y_test):
        '''
        This function calculates the accuracy of the model

        Parameters:
            mode: A model that is fitted
            X_test
            y_test
        '''
        y_pred = model.predict(X_test)

        # Measure accuracy
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Output the metrics
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R-squared:", r2)