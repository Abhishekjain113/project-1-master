import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.datascience.entity.config_entity import ModelEvaluationCOnfig
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories, save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationCOnfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Evaluate performance metrics: RMSE, MAE, and R2.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        """
        Log the model's performance into MLflow and save evaluation metrics.
        """
        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Prepare features and target
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column].to_numpy().ravel()  # Flatten to 1D

        # Set MLflow URI and start a new run
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.end_run()  # Ensure no active run is running before starting a new one

        with mlflow.start_run():
            # Predict target values
            predicted_qualities = model.predict(test_x).ravel()  # Flatten to 1D if necessary

            # Evaluate performance metrics
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics to JSON
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Log the model to MLflow
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")
