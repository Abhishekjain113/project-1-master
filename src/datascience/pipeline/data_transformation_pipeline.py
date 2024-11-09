from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_transformation import DataTransformation
from src.datascience import logger

from pathlib import Path

STAGE_NAME='Data transformtion stage'

class DataTransformationTraningPipeline:
    def __init__(self):
        pass
    def initiate_data_transformation(self):
        try:
            with open(Path('artifacts/data_validation/status.txt'),'r') as f:
                status=f.read().split(' ')[-1]
            if status=='True':
                config=ConfigurationManager()
                data_tranformation_config=config.get_data_transformation_config()
                data_tranformation=DataTransformation(config=data_tranformation_config)
                data_tranformation.train_test_splitting()
            else:
                raise Exception("data scheme is not valid")
        except Exception as e:
            print(e)