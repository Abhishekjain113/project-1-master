import os 
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from urllib.parse import urlparse
