from LoanApproval.logger import logging
from LoanApproval.exception import LoanApprovalException
from LoanApproval.entity.config_entity import ModelEvaluationConfig
from LoanApproval.entity.artifact_entity import ClassModelTrainerArtifact, DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, BaseDataTransformationArtifact
from LoanApproval.constant import *
import numpy as np
import os
import sys
from LoanApproval.util.util import write_yaml_file, read_yaml_file, load_object,load_data
from LoanApproval.entity.model_factory import evaluate_classification_model #, evaluate_regression_model




class ModelEvaluation:

    #def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ClassModelTrainerArtifact):
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_transformation_artifact: BaseDataTransformationArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ClassModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
    
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            #self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.data_validation_artifact = data_validation_artifact
    
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            print("========== model evaluation v1: model evaluation file path =============")
            print(model_evaluation_file_path)
            print("============================================================="*2)

            if not os.path.exists(model_evaluation_file_path):
                
                write_yaml_file(file_path=model_evaluation_file_path)

                return model
            
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path

            print("========== model evaluation v2: eval file path =============")
            print(eval_file_path)
            print("============================================================="*2)

            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path


            print("========== model evaluation v3: trained model file path =============")
            print(trained_model_file_path)
            print("============================================================="*2)


            trained_model_object = load_object(file_path=trained_model_file_path)

            #test_file_path = self.data_ingestion_artifact.test_file_path
            #train_file_path = self.data_ingestion_artifact.train_file_path

            train_file_path = self.data_transformation_artifact.transformed_resampled_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_non_resampled_test_file_path


            print("========== model evaluation v4: train and test file path =============")
            print(train_file_path)
            print(test_file_path)
            print("============================================================="*2)


            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path,
                                                           )
            test_dataframe = load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path,
                                                          )
            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # target_column
            logging.info(f"Converting target column into numpy array.")

            print("========= Model Evaluation ==========="*3)
            print("Converting target column into numpy array.")
            print("======================================"*3)

            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")


            print("========= Model Evaluation ==========="*3)
            print("Dropping target column from the dataframe.")
            print("======================================"*3)


            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")

                print("========= Model Evaluation ==========="*3)
                print("Not found any existing model. Hence accepting trained model")
                print("======================================"*3)


                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]

            #metric_info_artifact = evaluate_regression_model(model_list=model_list,
            #                                                   X_train=train_dataframe,
            #                                                   y_train=train_target_arr,
            #                                                   X_test=test_dataframe,
            #                                                   y_test=test_target_arr,
            #                                                   base_accuracy=self.model_trainer_artifact.model_accuracy,
            #                                                   )

            metric_info_artifact, _ = evaluate_classification_model(model_list=model_list,
                                                   X_train=train_dataframe,
                                                   y_train=train_target_arr,
                                                   X_test=test_dataframe,
                                                   y_test=test_target_arr,
                                                   base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                   )
            
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>' * 20} Model Evaluation log completed {'<' * 20} ")