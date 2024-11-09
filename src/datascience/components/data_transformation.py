import os
import pandas as pd

from src.datascience import logger
from sklearn.model_selection import train_test_split
from src.datascience.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
    
    def train_test_splitting(self):
        data=pd.read_csv(self.config.data_path)
        train,test=train_test_split(data,random_state=42)

        train.to_csv(os.path.join(self.config.root_dir,'train.csv'))
        test.to_csv(os.path.join(self.config.root_dir,'test.csv'))


        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
                