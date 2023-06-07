from LoanApproval.logger import logging
from LoanApproval.exception import LoanApprovalException
from LoanApproval.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact 
from LoanApproval.entity.config_entity import ModelPusherConfig
import os, sys
import shutil


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            logging.info(f"{'>>' * 30} Model Pusher log started {'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def export_model(self) -> ModelPusherArtifact:
        try:
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_pusher_config.export_dir_path
            
            model_file_name = os.path.basename(evaluated_model_file_path)
            export_model_file_path = os.path.join(export_dir, model_file_name)
            
            print("============= Model Pusher: ========="*5)
            print(evaluated_model_file_path)
            print(export_dir)
            print(model_file_name)
            print(export_model_file_path)
            print("==================="*4)

            ################### PATH PREPARATION FOR SCORE.CSV FILE COPYING INTO SAVED_MODEL FOLDER ##########
            head, _ = os.path.split(evaluated_model_file_path)
            head, _ = os.path.split(head)
            score_file_path_source = os.path.join(head,"score", "model_score.csv")
            export_score_dir = os.path.join(export_dir, "score")
            os.makedirs(export_score_dir, exist_ok=True)

            score_file_path_destination = os.path.join(export_score_dir, "model_score.csv")

            print("============= Model Pusher : Score_file_path ========="*5)
            print(score_file_path_source)
            print(score_file_path_destination)

            ##################################################################

            logging.info(f"Exporting model file: [{export_model_file_path}]")
            os.makedirs(export_dir, exist_ok=True)

            shutil.copy(src=evaluated_model_file_path, dst=export_model_file_path)

            shutil.copy(src=score_file_path_source, dst=score_file_path_destination)    # COPY Score file too
            shutil.copy(src=score_file_path_source, dst=os.path.join(export_score_dir, "model_score.html"))    # COPY Score Plot file too



            # we can call a function to save model to Azure blob storage/ google cloud strorage / s3 bucket
            
            logging.info(f"Trained model: {evaluated_model_file_path} is copied in export dir:[{export_model_file_path}]")

            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path=export_model_file_path)
            
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            
            return model_pusher_artifact
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20} Model Pusher log completed {'<<' * 20} ")