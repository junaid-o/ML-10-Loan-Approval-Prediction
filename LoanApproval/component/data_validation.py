import shutil
import natsort
from LoanApproval.logger import logging
from LoanApproval.exception import LoanApprovalException
from LoanApproval.entity.config_entity import DataValidationConfig
from LoanApproval.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.tests import *

import pandas as pd
import os, sys
import json


class DataValidation:

    def __init__(self, data_validation_config:DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'='*20} DATA VALIDATION LOG STARTED. {'='*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            
            #self.training_file_path = "LoanApproval\\dataset_base\\Processed_Dataset\\Resampled_Dataset\\train_resampled\\train_resample_major.csv"
            #self.testing_file_path = "LoanApproval\\dataset_base\\Processed_Dataset\\Resampled_Dataset\\test_resampled\\test_non_resample_major.csv"
            #self.report_file_path = "LoanApproval\\artifact\\data_ingestion\\report\\report.json"
            #self.report_page_path = "LoanApproval\\artifact\\data_ingestion\\report\\report.html"
        except Exception as e:
            raise LoanApprovalException(e,sys) from e
        

    def get_train_and_test_df(self):
        try:            
            
            training_file_path = self.data_ingestion_artifact.train_file_path
            testing_file_path = self.data_ingestion_artifact.test_file_path

            print("==== Validation:  training file path ===="*4)
            print(f"{training_file_path}")
            print("=================="*4)
            
            train_df = pd.read_csv(training_file_path)
            test_df = pd.read_csv(testing_file_path)
           
            return train_df, test_df
        
        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        
    def is_train_test_file_exists(self) -> bool:
        try:
            logging.info(f"Checking if train and test csv file is available")
            is_train_file_exists = False
            is_test_file_exists = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            print("==== Validation 2:  training file path ===="*4)
            print(f"{train_file_path}")
            print("=================="*4)

            #train_file_path = self.training_file_path
            #test_file_path = self.testing_file_path

            is_train_file_exists = os.path.exists(train_file_path)
            is_test_file_exists = os.path.exists(test_file_path)
            
            is_available = is_train_file_exists and is_test_file_exists

            logging.info(f"Is train adn test file exists? --> {is_available}")

            if not is_available:
                #training_file = self.training_file_path
                #testing_file = self.testing_file_path

                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path                

                message = f"Training file [{training_file}] or Testing file [{testing_file}] is not present"
                
                logging.info(message)
                raise Exception(message)

            
            return is_available
        except Exception as e:
            raise LoanApprovalException(e, sys) from e    
        
    def validate_dataset_schema(self) -> bool:
        try:
            validation_status = False
            #validate training and testing dataset using schema file
                #1. Number of Column
                #2. Check the value of ocean proximity                 
                # acceptable values        
                #3. Check column names
            
            validation_status = True
            
            return validation_status
        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        
    def get_and_save_data_drift_report(self):

        try:           
                        
            ########### NEW METHOD ###############
            train_df, test_df = self.get_train_and_test_df()
            
            print("===== TRAIN DF COLUMNS ======"*5)
            print(train_df.columns)
            print(train_df.shape)

            print("===== TEST DF COLUMNS ======"*5)
            print(test_df.columns)


            report = Report(metrics=[ DataDriftPreset(),])
            report.run(reference_data=train_df, current_data=test_df)

            ########### SAVING JSON FILE ########
            report_file_path = self.data_validation_config.report_file_path
            
            #report_file_path = self.report_file_path
            
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)            
            
            report.save_json(filename= report_file_path)

            ############## SAVING HTML FILE #########
            report_page_file_path = self.data_validation_config.report_page_file_path
            
            #report_page_file_path = self.report_page_path
            
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)            
            

            report.save_html(filename= report_page_file_path)

            print('====================='*5)
            print("test.......\n"*10)
            
            
            ##################  COPYING DRIFT REPORT TO TEMPLATES FOLDER  #########################

            def get_copy_of_drift_report():
                try:
                    ####################### COPYING PROFILE_REPORT_2 TO TEMPLATE #######################################
                    src_dir = os.path.join("LoanApproval\\artifact", "data_validation")
                    dest_dir = os.path.join("templates")                    
                    # Get the list of folders in the source directory
                    folders = natsort.natsorted(os.listdir(src_dir))
                    # Get the most recent folder
                    most_recent_folder = folders[-1]
                    # Construct the path to the most recent folder
                    most_recent_folder_path = os.path.join(src_dir, most_recent_folder)
                    # Get the list of files in the most recent folder
                    files = natsort.natsorted(os.listdir(most_recent_folder_path))
                    # Get the most recent html file not json
                    #most_recent_file = files[1]
                    most_recent_file = "drift_report.html"
                    
                    # Construct the path to the most recent file
                    most_recent_file_path = os.path.join(most_recent_folder_path, most_recent_file)
                    # Copy the most recent file to the destination directory
                    shutil.copy(most_recent_file_path, dest_dir)                              
                except Exception as e:
                    raise LoanApprovalException(e, sys) from e            
                
            #########################################################
            get_copy_of_drift_report()

            ########################################################
            
            return report

        except Exception as e:
            raise LoanApprovalException(e, sys) from e        
        
    def save_data_drift_report_page(self):
        try:
            ######### DEPRICATED METHOD ########
            #dashboard = Dashboard(tabs=[DataDriftTab()])
            #train_df, test_df = self.get_train_and_test_df()
            #dashboard.calculate(train_df, test_df)
            ################################

            #report_page_file_path = self.data_validation_config.report_page_file_path
            #report_page_dir = os.path.dirname(report_page_file_path)
            #os.makedirs(report_page_dir, exist_ok=True)

            #dashboard.save(report_page_file_path)
            
            #report.save_html(filename= report_file_path)
            pass
            
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def is_data_drift_found(self) -> bool:

        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()

            return True
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.is_train_test_file_exists()            
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(schema_file_path= self.data_validation_config.schema_file_path,
                                                              report_file_path= self.data_validation_config.report_file_path,
                                                              report_page_file_path= self.data_validation_config.report_page_file_path,
                                                              is_validated= True,
                                                              message= "Data Validation Performed Sucessfully")
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            print("====="*40)
            print(data_validation_artifact)
            print('===='*40)

            return data_validation_artifact
        except Exception as e:
            raise LoanApprovalException(e, sys) from e 


    def __del__(self):
        logging.info(f"{'>>'*30} Data Validation log completed {'<<'*30} \n\n")