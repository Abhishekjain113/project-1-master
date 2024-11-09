from src.datascience.pipeline.data_validation_pipeline import DataValidationTraningPipeline
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTraningPipeline

from src.datascience import logger

SATGE_NAME='Data ingestion stage'
try:
    logger.info(f'stage {SATGE_NAME} started')
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f'{SATGE_NAME} compelted')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataValidationTraningPipeline()
   data_ingestion.initiate_data_validation()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataTransformationTraningPipeline()
   data_ingestion.initiate_data_transformation()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
