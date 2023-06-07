import os
import sys
from LoanApproval.logger import logging
from LoanApproval.exception import LoanApprovalException
from LoanApproval.util.util import load_object

import pandas as pd


class LoanApprovalData:

    def __init__(self, gender: int, married: int, education: int, self_employed: int, credit_history: int,
                 property_area, Dependents, ApplicantIncome: float,
                 CoapplicantIncome: float, LoanAmount: float, Loan_Amount_Term
                 ):

        logging.info(f"{'>>' * 30} LoanApprovalData log started {'<<' * 30} ")

        try:
            # Define a dictionary to map property area values to variables
            property_area_mapping = {'Urban': [1, 0, 0],
                        'Semiurban': [0, 1, 0],
                        'Rural': [0, 0, 1]
                        }

            # Process the extracted values
            Property_Area_Urban, Property_Area_Semiurban, Property_Area_Rural = property_area_mapping.get(property_area, [0, 0, 0])

            dependent_mapping = {'Dependent_1': [1, 0, 0],
                                    'Dependent_2': [0, 1, 0],
                                    'Dependent_3': [0, 0, 1]
                                    }
            Dependents_1, Dependents_2, Dependents_3 = dependent_mapping.get(Dependents, [0, 0, 0])

        except Exception as e:
            raise LoanApprovalException(e, sys) from e



        try:
            self.Gender_Male = gender
            self.Married_Yes = married
            self.Education_Not_Graduate = education            
            self.Self_Employed_Yes = self_employed
            self.Credit_History = credit_history
            #self.property_area = property_area
            self.Property_Area_Semiurban = Property_Area_Semiurban
            self.Property_Area_Urban = Property_Area_Urban
            self.Dependents_1 = Dependents_1
            self.Dependents_2 = Dependents_2
            self.Dependents_3_plus = Dependents_3
            self.ApplicantIncome = ApplicantIncome
            self.CoapplicantIncome = CoapplicantIncome
            self.LoanAmount = LoanAmount
            self.Loan_Amount_Term= Loan_Amount_Term

        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def get_LoanApproval_input_data_frame(self):

        try:
            logging.info(f"Converting to DataFrame")
            LoanApproval_input_dict = self.get_LoanApproval_data_as_dict()
            print("==========get_LoanApproval_input_data_frame=========="*2)
            print(pd.DataFrame(LoanApproval_input_dict).columns)

            return pd.DataFrame(LoanApproval_input_dict)

        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def get_LoanApproval_data_as_dict(self):
        try:
            logging.info(f"getting LoanApproval data as_dict")

            input_data = {"Credit_History":[self.Credit_History],
                          "Gender_Male": [self.Gender_Male],                          
                          "Married_Yes":[self.Married_Yes],
                          "Dependents_1":[self.Dependents_1],
                          "Dependents_2":[self.Dependents_2],
                          "Dependents_3_plus":[self.Dependents_3_plus],                          
                          "Education_Not_Graduate":[self.Education_Not_Graduate],
                          "Self_Employed_Yes":[self.Self_Employed_Yes],
                          "Property_Area_Semiurban":[self.Property_Area_Semiurban],
                          "Property_Area_Urban":[self.Property_Area_Urban],
                          "ApplicantIncome":[self.ApplicantIncome],
                          "CoapplicantIncome":[self.CoapplicantIncome],
                          "LoanAmount":[self.LoanAmount],
                          "Loan_Amount_Term": [self.Loan_Amount_Term],
                          }
           
            return input_data
        except Exception as e:
            raise LoanApprovalException(e, sys)


class LoanApprovalPredictor:

    def __init__(self, model_dir: str):
        try:
            logging.info(f"{'>>' * 30} LoanApprovalPredictor log started {'<<' * 30} ")

            self.model_dir = model_dir
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def get_latest_model_path(self):
        try:
            logging.info(f"getting latest model path")

            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)

            logging.info(f"latest model path: [ {latest_model_path} ]")
            
            return latest_model_path
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def predict(self, X):
        try:
            logging.info(f"LoanApprovalPredictor is Making Predictions")

            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            
            logging.info(f"Model objct loaded from path: [ {model_path} ]")

            class_mapping = {0: 'Not_Eligible',
                                   1: 'Eligible',
                                  }
            
            class_prediction = class_mapping[model.predict(X)[0]]

            logging.info(f"Predictions: [ {class_prediction} ]")

            return class_prediction

        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        
    def bulk_prediction(self, X):
        try:
            logging.info(f"LoanApprovalPredictor is Making Predictions")

            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            
            logging.info(f"Model objct loaded from path: [ {model_path} ]")

           

            class_mapping = {0: 'Not_Eligible',
                                   1: 'Eligible',
                                   }
            
            class_prediction = model.predict(X)
            
            logging.info(f"Predictions: [ {class_prediction} ]")
            # returning original prdictions without mapping but it will be don in app.py file
            return class_prediction

        except Exception as e:
            raise LoanApprovalException(e, sys) from e        