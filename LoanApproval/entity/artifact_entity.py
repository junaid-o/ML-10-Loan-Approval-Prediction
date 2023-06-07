from collections import namedtuple

DataIngestionArtifact =  namedtuple("DataIngestionArtifact",
                                    ["train_file_path", "test_file_path","is_ingested","message"])

BaseDataIngestionArtifact =  namedtuple("BaseDataIngestionArtifact",
                                    ["train_file_path_base", "test_file_path_base","is_ingested","message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["schema_file_path", "report_file_path", "report_page_file_path", "is_validated", "message"])

BaseDataTransformationArtifact = namedtuple("BaseDataTransformationArtifact",
                                        ["is_transformed", "message", "transformed_resampled_train_file_path",
                                         "transformed_non_resampled_test_file_path"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",
                                        ["is_transformed", "message", "transformed_train_file_path",
                                         "transformed_test_file_path", "preprocessed_object_file_path"])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",
                                  ["is_trained", "message", "trained_model_file_path",
                                   "train_rmse", "test_rmse", "train_accuracy", "test_accuracy","model_accuracy"])

ClassModelTrainerArtifact = namedtuple("ModelTrainerArtifact",
                                  ["is_trained", "message", "trained_model_file_path",
                                   "train_f1_weighted","test_f1_weighted",
                                   "train_balanced_accuracy", "test_balanced_accuracy","model_accuracy"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact",["is_model_accepted", "evaluated_model_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pusher", "export_model_file_path"])