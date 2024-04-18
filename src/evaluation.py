import logging
from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation of our models
    """
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates the scores of the model
        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        Returns:
            None
        """
        pass


class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE : {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e


class R2(Evaluation):
     """
     Evaluation Strategy that uses R2 Score
     """
     def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2: {}".format(r2))
            return r2
        except Exception as e:
            logging.info("Error in calculating r2: {}".format(e))
            raise e
        
             
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses RMSE score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.info("Error in calculating RMSE: {}".format(e))
            raise e