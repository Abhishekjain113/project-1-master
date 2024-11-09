from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_validation import DataValidation
from src.datascience import logger

STAGE_NAME='Data validation stage'

class DataValidationTraningPipeline:
    def __init__(self):
        pass
    def initiate_data_validation(self):
        config=ConfigurationManager()
        data_validation_config=config.get_data_validation_config()
        data_validation=DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()
  
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTraningPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
          