from collections import namedtuple


# A tuple in which each element has a name referred as named tuple

DataIngestionConfig = namedtuple("DataIngestionConfig", ["dataset_download_url",
                                                          "tgz_download_dir",
                                                           "raw_data_dir",
                                                           "ingested_train_dir",
                                                           "ingested_test_dir"])

BaseDataIngestionConfig = namedtuple("BaseDataIngestionConfig", ["raw_data_dir",
                                                          "processed_data_dir",                                                           
                                                           "cleaned_data_dir",
                                                           ])

BaseDataTransformationConfig = namedtuple("BaseDataTransformationConfig",[ "resampled_data_dir",
                                                                          "train_resampled_dir",
                                                                          "test_non_resampled_dir",
                                                                          "transformed_data_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path", "report_file_path", "report_page_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_dir",
                                                                   "transformed_test_dir",
                                                                   "preprocessed_object_file_path"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path", "base_accuracy", "model_config_file_path"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_file_path", "time_stamp"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])