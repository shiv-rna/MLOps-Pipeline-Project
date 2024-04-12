import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train: Training Data
            y_train: Training Labels

        Retunrs:
            None
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model

        Args:
            X_train: Training Data
            y_train: Training Labels

        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error("Error in Training the model: {}".format(e))
            raise e




