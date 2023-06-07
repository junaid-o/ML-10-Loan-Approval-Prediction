from sklearn.model_selection import train_test_split
from LoanApproval.exception import LoanApprovalException
from LoanApproval.logger import logging
from LoanApproval.entity.config_entity import BaseDataIngestionConfig, BaseDataTransformationConfig, DataTransformationConfig 
from LoanApproval.entity.artifact_entity import BaseDataIngestionArtifact, BaseDataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from LoanApproval.constant import *
from LoanApproval.util.util import read_yaml_file, save_object, save_numpy_array_data, load_data

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import dill
import sys,os
from cgi import test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer


class DataTransformation:

    def __init__(self, data_transformation_config: BaseDataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact, base_data_ingestion: BaseDataIngestionConfig):
    
        try:
            logging.info(f"{'>>' * 30} Data Transformation log started {'<<' * 30} ")
    
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            #self.data_validation_artifact = data_validation_artifact
            
            self.processed_data_dir_path = base_data_ingestion.processed_data_dir
            #self.processed_data_dir_path = "LoanApproval/dataset_base/Processed_Dataset"

            self.cleaned_data_dir = base_data_ingestion.cleaned_data_dir
            
        except Exception as e:
            raise LoanApprovalException(e,sys) from e


    def get_resampled_data(self):
        try:
            logging.info("Importing Train and Test files for Resampling of Data")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train = pd.read_csv(train_file_path)
            test = pd.read_csv(test_file_path)

            print("========= DataTransformation V1: train.columns =========")
            print(train.columns)
            print("================"*5)
                      
            X_train = train.drop(['Loan_Status_Y'], axis=1)
            y_train = train["Loan_Status_Y"]
            
            X_test = test.drop(['Loan_Status_Y'], axis=1)
            y_test = test["Loan_Status_Y"]

            X_train_dropped = X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']]
            X_test_dropped = X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']]


            fitted_scaler = StandardScaler().fit(X_train_dropped)
            #fitted_scaler = Normalizer().fit(X_train_dropped)

            X_train_scaled = fitted_scaler.transform(X_train_dropped)
            X_test_scaled = fitted_scaler.transform(X_test_dropped)

            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns= ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])

            X_train_scaled_final  = pd.concat([X_train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1).reset_index(drop=True), X_train_scaled_df, y_train],axis=1)
            X_test_scaled_final  = pd.concat([X_test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1).reset_index(drop=True), X_test_scaled_df, y_test],axis=1)


            #################### Rename the 'old_column' to 'new_column' #######################################
            X_train_scaled_final = X_train_scaled_final.rename(columns={'Dependents_3+': 'Dependents_3_plus',
                                                                         "Education_Not Graduate":"Education_Not_Graduate"})
            X_test_scaled_final = X_test_scaled_final.rename(columns={'Dependents_3+': 'Dependents_3_plus',
                                                                      "Education_Not Graduate":"Education_Not_Graduate"})

            ################## Datatype ###########################
            # Specify the desired data types as a dictionary
            dtypes = {'Gender_Male': int,
                    'Married_Yes': int,
                    'Education_Not_Graduate': int,
                    'Self_Employed_Yes': int,
                    'Credit_History': int,
                    'Property_Area_Semiurban': int,
                    'Property_Area_Urban': int,
                    'Dependents_1': int,
                    'Dependents_2': int,
                    'Dependents_3_plus': int,
                    'ApplicantIncome': float,
                    'CoapplicantIncome': float,
                    'LoanAmount': float}

            # Change the data types of the columns
            X_train_scaled_final = X_train_scaled_final.astype(dtypes)
            X_test_scaled_final = X_test_scaled_final.astype(dtypes)


            #####################################################


            logging.info("Exporting Resampled Train data.....")

            train_scaled_data_dir = os.path.join(self.data_transformation_config.resampled_data_dir, "train")

            print("========= DataTransformation V2: train_resample_dir =========")
            print(train_scaled_data_dir)
            print("================"*5)


            os.makedirs(train_scaled_data_dir, exist_ok=True)            
            train_scaled_file_path = os.path.join(train_scaled_data_dir, "train_cleaned_scaled.csv")            
            X_train_scaled_final.to_csv(train_scaled_file_path, index=False)
            
            logging.info(f"Non Resampled train file exportd: [ {train_file_path} ]")

            
            test_scaled_dir = os.path.join(self.data_transformation_config.resampled_data_dir , "test")
            os.makedirs(test_scaled_dir, exist_ok=True)
            
            test_scaled_file_path = os.path.join(test_scaled_dir, "test_cleaned_scaled.csv")
            
            X_test_scaled_final.to_csv(test_scaled_file_path, index=False)

            logging.info(f"Non-Resampled test file copied to: [ {test_scaled_file_path} ]")

            
            data_transformation_artifact = BaseDataTransformationArtifact(is_transformed=True, message="Data transformation successfull.",
                                                                          transformed_resampled_train_file_path = train_scaled_file_path,
                                                                          transformed_non_resampled_test_file_path= test_scaled_file_path)

            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")

            return data_transformation_artifact

        except Exception as e:
            raise LoanApprovalException(e,sys) from e

    def initiate_data_transformation(self):
        try:
            return self.get_resampled_data()
        
        except Exception as e:
            raise LoanApprovalException(e,sys) from e
    

    def __del__(self):
        logging.info(f"{'>>'*30} Data Transformation log completed {'<<'*30} \n\n")