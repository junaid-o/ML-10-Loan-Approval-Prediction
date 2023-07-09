import importlib
import numpy as np
import os
import sys

import pandas as pd
import yaml

from pyexpat import model
from cmath import log
from typing import List
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, roc_auc_score, log_loss

from collections import namedtuple
from LoanApproval.logger import logging
from LoanApproval.exception import LoanApprovalException
from LoanApproval.entity.config_entity import ModelTrainerConfig


GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"


InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel",
                                   ["model_serial_number", "model", "best_model", "best_parameters", "best_score",])

BestModel = namedtuple("BestModel",
                       ["model_serial_number", "model", "best_model", "best_parameters", "best_score", ])

#MetricInfoArtifact = namedtuple("MetricInfoArtifact",
#                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy", "test_accuracy", "model_accuracy", "index_number"])#

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_f1_weighted", "test_f1_weighted",
                                 "train_balanced_accuracy", "test_balanced_accuracy",                                 
                                  "model_accuracy", "index_number"])



def evaluate_classification_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy: float= "base_accuracy")->MetricInfoArtifact:
    """
    Description:
    This function compare multiple classification model return best model

    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature

    return
    It retured a named tuple
    
    ClassMetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_f1_weighted", "test_f1_weighted",
                                 "train_balanced_accuracy", "test_balanced_accuracy",                                 
                                  "model_accuracy", "index_number"])

    """
    try:
        
        f1_train_list = []
        f1_test_list = []
        precision_train_list = []
        precision_test_list = []
        balanced_accuracy_train_list = []
        balanced_accuracy_test_list = []
        balanced_accuracy_diff_list = []
        log_loss_train_list = []
        log_loss_test_list = []
        roc_auc_ovr_weighted_test_list = []
        roc_auc_ovr_weighted_train_list= []
        model_name_list = []

        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            model_name_list.append(type(model).__name__)

            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
            
            #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculating F1_weighted score on training and testing dataset
            train_f1 = f1_score(y_train, y_train_pred, average="weighted")
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")

            f1_train_list.append(train_f1)
            f1_test_list.append(test_f1)


            print("=======y_test shape======"*5)
            print(y_train.shape)
            print(y_train)

            # Precision Calculcation
            precision_train = precision_score(y_train, model.predict(X_train), zero_division=0, average='weighted')
            precision_test = precision_score(y_train, model.predict(X_train), zero_division=0, average='weighted')

            precision_train_list.append(precision_train)
            precision_test_list.append(precision_test)

            # Calculating roc_auc_ovr_weighted
            roc_auc_ovr_weighted_train =  roc_auc_score(y_train, model.predict(X_train), multi_class='ovr', average='weighted')
            roc_auc_ovr_weighted_test =  roc_auc_score(y_test, model.predict(X_test), multi_class='ovr', average='weighted')
            roc_auc_ovr_weighted_train_list.append(roc_auc_ovr_weighted_train)
            roc_auc_ovr_weighted_test_list.append(roc_auc_ovr_weighted_test)

            # Calculating Balanced Accuracy on training and testing dataset
            train_balanced_accuracy_score = balanced_accuracy_score(y_train, y_train_pred)
            test_balanced_accuracy_score = balanced_accuracy_score(y_test, y_test_pred)

            balanced_accuracy_train_list.append(train_balanced_accuracy_score)
            balanced_accuracy_test_list.append(test_balanced_accuracy_score)
            
            ####################################################################################
            # Calculating harmonic mean of train_accuracy and test_accuracy
            # Fowlkes-Mallows index. It is a measure of similarity between two sets of clustering or classification results,
            #  where the two sets of results are compared against a reference (or ground truth) clustering or classification.
            #  Specifically, the Fowlkes-Mallows index combines measures of precision and recall to give an overall similarity
            #  score between the two sets of results.
            #  The Fowlkes-Mallows index is a weighted harmonic mean of precision and recall.

            model_accuracy = (2 * (train_balanced_accuracy_score * test_balanced_accuracy_score)) / (train_balanced_accuracy_score + test_balanced_accuracy_score)
            
            diff_test_train_acc = abs(test_balanced_accuracy_score - train_balanced_accuracy_score)
            balanced_accuracy_diff_list.append(diff_test_train_acc)


            loss_train = log_loss(y_train, model.predict_proba(X_train))
            loss_test = log_loss(y_test, model.predict_proba(X_test))
            log_loss_test_list.append(loss_test)
            log_loss_train_list.append(loss_train)            


            print("============ MODEL Factory:LOG LOSS=========="*2)
            print("loss_train: ", loss_train)
            print("loss_test: ", loss_test)
            print("================================"*2)
            
            
            ################################################################

            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train F1_weighted\t\t Test F1_weighted\t\t   roc_auc_ovr_weighted_train    \t\t roc_auc_ovr_weighted_train \t\t Train Balanced Accuracy Score\t\t Test Balanced Accuracy Score\t\t Average Score")
            logging.info(f"{train_f1}\t\t {test_f1}\t\t {roc_auc_ovr_weighted_train}\t\t {roc_auc_ovr_weighted_test}\t\t\t {train_balanced_accuracy_score}\t\t {test_balanced_accuracy_score} \t\t\t {model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].")
            logging.info(f"Log Loss Train: [{loss_train}].")
            logging.info(f"Log Loss Test: [{loss_test}].")
            

            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model

            logging.info(f"train balanced accuracy: {train_balanced_accuracy_score} and model_accuracy(average): {model_accuracy} and base_accuracy: {base_accuracy} and diff_test_train_accuracy{diff_test_train_acc}")
            
            precision_logic = (precision_train >= 7) and abs(precision_train -  precision_test) <=0.3
            f1_logic = (train_f1 >= 0.40) and abs(train_f1 - test_f1) <= 0.5
            roc_auc_logic = (roc_auc_ovr_weighted_train >= 0.4) and abs(roc_auc_ovr_weighted_train - roc_auc_ovr_weighted_test) <= 0.5
            model_accuracy_logic = (train_balanced_accuracy_score >= base_accuracy) and diff_test_train_acc <= 0.6
            loss_logic = (loss_train <= 1.015) and abs(loss_train - loss_test) <= 1
            
            #if model_accuracy >= base_accuracy and diff_test_train_acc < 0.5:
            if  precision_logic and f1_logic and roc_auc_logic and model_accuracy_logic: #and loss_logic:
            
                #base_accuracy = model_accuracy
                
                print('='*20)
                print("TEST_1: MODEL IS ABOVE BASE ACCURACY")
                print('='*20)

                logging.info(f"model_accuracy: {model_accuracy} and base_accuracy: {base_accuracy} and diff_test_train_accuracy{diff_test_train_acc}")
                
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                          model_object=model,
                                                          train_f1_weighted=train_f1,
                                                          test_f1_weighted=test_f1,
                                                          train_balanced_accuracy = train_balanced_accuracy_score,
                                                          test_balanced_accuracy= test_balanced_accuracy_score,                                                         
                                                          model_accuracy=model_accuracy,
                                                          index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")

            index_number += 1            
        if metric_info_artifact is None:

            print('='*20)
            print("TEST_2: MODEL NOT ACCEPTED")

            logging.info(f"No model found with higher accuracy than base accuracy")

        #################################################################

        data = {"models": model_name_list,
                "Precision_train": precision_train_list,
                "Precision_test": precision_test_list,
                "f1_weighted_train": f1_train_list,
                "f1_weighted_test": f1_test_list,
                "roc_auc_ovr_weighted_train": roc_auc_ovr_weighted_train_list,
                "roc_auc_ovr_weighted_test": roc_auc_ovr_weighted_test_list,
                "balanced_accuracy_train": balanced_accuracy_train_list,
                "balanced_accuracy_test": balanced_accuracy_test_list,
                "balanced_accuracy_diff": balanced_accuracy_diff_list,
                "log_loss_train": log_loss_train_list,
                "log_loss_test": log_loss_test_list}
        df_scores = pd.DataFrame(data= data)

        #scores_path = os.makedirs(os.path.join(ModelTrainerConfig.trained_model_file_path, "scores"), exist_ok=True)
        #df_scores.to_csv(os.path.join(scores_path, "scores.csv"), index=False)
        ################################################################


        return metric_info_artifact,df_scores
        
    except Exception as e:
        raise LoanApprovalException(e, sys) from e


def get_sample_model_config_yaml_file(export_dir: str):
    try:
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,
                    "verbose": 1
                }

            },
            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY:
                        {"param_name1": "value1",
                         "param_name2": "value2",
                         },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }

                },
            }
        }
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise LoanApprovalException(e, sys)


class ModelFactory:
    def __init__(self, model_config_path: str = None,):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)

            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            
            print(property_data)

            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    @staticmethod
    def class_for_name(module_name:str, class_name:str):
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearchCV class
            
           
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name
                                                             )

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)

            
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)

            grid_search_cv.fit(input_feature, output_feature)

            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )
            
            return grid_searched_best_model
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:

        """
        This function will return a list of model details.
        return List[ModelDetail]
        """
        
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():

                model_initialization_config = self.models_initialization_config[model_serial_number]

                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY])
                model = model_obj_ref()
                
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)

                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list

            return self.initialized_model_list
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail, input_feature, output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:

        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(initialized_model = initialized_model_list,
                                                                                                     input_feature=input_feature,
                                                                                                     output_feature=output_feature)
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel], base_accuracy= "base_accuracy") -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def get_best_model(self, X, y, base_accuracy = "base_accuracy") -> BestModel:

        try:
            logging.info("Started Initializing model from config file")

            initialized_model_list = self.get_initialized_model_list()

            logging.info(f"Initialized model: {initialized_model_list}")
            
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(initialized_model_list=initialized_model_list,
                                                                                                       input_feature=X,
                                                                                                       output_feature=y
                                                                                                       )
            
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise LoanApprovalException(e, sys)