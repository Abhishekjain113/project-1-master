from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline

SATGE_NAME='Data ingestion stage'
try:
    logger.info(f'stage {SATGE_NAME} started')
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f'{SATGE_NAME} compelted')
except Exception as e:
    logger.exception(e)
    raise e