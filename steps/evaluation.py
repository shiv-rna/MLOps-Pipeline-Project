import logging
from typing import Tuple
from typing_extensions import Annotated
import mlflow

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, RMSE, R2

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    """
    Evaluate the model on the ingested data.

    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2_score, rmse
    
    except Exception as e:
        logging.error("Error evaluating the model: {}".format(e))
        raise e