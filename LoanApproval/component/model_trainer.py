import os
import plotly.graph_objects as go
import pandas as pd
from LoanApproval.exception import LoanApprovalException
import sys
from LoanApproval.logger import logging
from typing import List
from LoanApproval.entity.artifact_entity import BaseDataTransformationArtifact, ClassModelTrainerArtifact, DataTransformationArtifact #, ModelTrainerArtifact
from LoanApproval.entity.config_entity import ModelTrainerConfig
from LoanApproval.util.util import load_numpy_array_data,save_object,load_object
from LoanApproval.entity.model_factory import MetricInfoArtifact, ModelFactory,GridSearchedBestModel
from LoanApproval.entity.model_factory import evaluate_classification_model



class LoanApprovalEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        #transformed_feature = self.preprocessing_object.transform(X)
        #return self.trained_model_object.predict(transformed_feature)

        return self.trained_model_object.predict(X)

    def predict_proba(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        #transformed_feature = self.preprocessing_object.transform(X)
        #return self.trained_model_object.predict(transformed_feature)

        return self.trained_model_object.predict_proba(X)


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: BaseDataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def initiate_model_trainer(self) -> ClassModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_resampled_train_file_path

            print('======== Model Trainer V1: transformed_train_file_path ======='*2)
            print(transformed_train_file_path)
            print("================================================================"*2)

            #train_array = load_numpy_array_data(file_path=transformed_train_file_path)

            train = pd.read_csv(transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")

            transformed_test_file_path = self.data_transformation_artifact.transformed_non_resampled_test_file_path
            
            #test_array = load_numpy_array_data(file_path=transformed_test_file_path)
            test = pd.read_csv(transformed_test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            x_train,y_train,x_test,y_test = train.drop(["Loan_Status_Y"], axis=1),train["Loan_Status_Y"],test.drop(["Loan_Status_Y"], axis=1),test["Loan_Status_Y"]
            

            print('======== Model Trainer V2: x_train data info ======='*2)
            print(x_train.shape)
            print(x_train.columns)
            print("================================================================"*2)            


            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)
            

            print("========== model trainer: Model Factory =============")
            print("Model factory done! going to base_accuracy")
            print("============================================================="*2)            

            base_accuracy = self.model_trainer_config.base_accuracy

            logging.info(f"Expected accuracy: {base_accuracy}")

            print("========== model trainer: Model Factory base accuracy =============")
            print("Model factory done! going to base_accuracy")
            print("============================================================="*2)  

            logging.info(f"Initiating operation model selection")
            best_model = model_factory.get_best_model(X=x_train,y=y_train, base_accuracy=base_accuracy)
            
            logging.info(f"Best model found on training dataset: {best_model}")
            
            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list:List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list
            
            model_list = [model.best_model for model in grid_searched_best_model_list ]
            logging.info(f"Evaluating all trained model on training and testing dataset both")


            print('======== Model Trainer V3:Model List ======='*2)
            print(model_list)
            print("================================================================"*2)            



            ############################################# MODEL EVALUATION #######################################################################

            #metric_info:MetricInfoArtifact = evaluate_regression_model(model_list=model_list,X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)

            metric_info: MetricInfoArtifact = evaluate_classification_model(model_list=model_list,X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)

            metric_info, model_scores = metric_info

            print(model_scores)
            ####################################################################################################################

            logging.info(f"Best found model on both training and testing dataset.")

            #preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            #preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            #preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)

            model_object = metric_info.model_object


            trained_model_file_path= self.model_trainer_config.trained_model_file_path

            print('======== Model Trainer: trained model file path ======='*2)
            print(trained_model_file_path)
            print("================================================================"*2)                        

            LoanApproval_model = LoanApprovalEstimatorModel(preprocessing_object=None, trained_model_object=model_object)

            logging.info(f"Saving model at path: {trained_model_file_path}")
            

            save_object(file_path=trained_model_file_path, obj=LoanApproval_model)
            

            print("=========    MODEL_TRAINER: LoanApproval MODEL")
            print(LoanApproval_model)
            print("===="*5)
            
            #################### Exporting SCORE CSV FILE ################

            head, tail= os.path.split(trained_model_file_path)

            head, _ = os.path.split(head)

            model_scores_file_dir = os.path.join(head, "score")
            os.makedirs(model_scores_file_dir)
            model_scores_file_path = os.path.join(model_scores_file_dir, "model_score.csv")
            model_scores.to_csv(model_scores_file_path, index=False)
            
            ##################### EXPORTING PLOT OF SCORES #######################
            df = model_scores
            df = df.set_index('models')
            print(df.head())
            df1 = df[['f1_weighted_train','roc_auc_ovr_weighted_train','balanced_accuracy_train',"log_loss_train"]]
            df2 = df[['f1_weighted_test','roc_auc_ovr_weighted_test','balanced_accuracy_test','log_loss_test']]
        

            def multi_plot(df1,df2, title, addAll = True):
                
                fig = go.Figure()
                

                for column1 in df1.columns.to_list():
                    f1= fig.add_trace(
                        go.Bar(
                            x = df1.index,
                            y = df1[column1],
                            name = column1,
                        )
                    )

                for column2 in df2.columns.to_list():
                    fig.add_trace(
                        go.Bar(
                            x = df2.index,
                            y = df2[column2],
                            name = column2
                        )
                    )


                button_all = [dict(label = 'Train',
                                method = 'update',
                                args = [{'visible': df1.columns.isin(df1.columns),
                                        'title': 'All',
                                        'showlegend':True}]),
                            dict(label = 'Select',
                                method = 'update',
                                args = [{'visible': df2.columns.isin(df2.columns),
                                        'title': 'All',
                                        'showlegend':True}]),
                            
                            ]
                            
                            

                def create_layout_button(column1):
                    return dict(label = column1,
                                method = 'update',
                                args = [{'visible': df1.columns.isin([column1]),
                                        'title': column1,
                                        'showlegend': True}])
                
                def create_layout_button2(column2):
                    return dict(label = column2,
                                method = 'update',
                                args = [{'visible': df2.columns.isin([column2]),
                                        'title': column2,
                                        'showlegend': True}]
                            )
                # Update remaining layout properties

                fig.update_layout(
                    updatemenus=[
                    
                        
                        go.layout.Updatemenu(
                        active = 0,
                        visible=True,
                        buttons = ([button_all[1]] * addAll) + list(df2.columns.map(lambda column: create_layout_button2(column))),
                        
                        direction="right",
                        pad={"r": 5, "t": 5,"l":5},
                        showactive=True,
                        x=-0.03,
                        xanchor="left",
                        y=1.1,
                        yanchor="bottom"),
                        
                        
                        
                    ],
                    yaxis_type="log"       
                )
                # Update remaining layout properties
                fig.update_layout(
                    title_text=title,
                    title_y=0.96,
                    
                    height=400,
                    #width = 1000,
                    showlegend=True,
                    legend=dict(yanchor="bottom",
                                            y=-0.5,
                                            xanchor="center",
                                            x=0.5,
                                            orientation='h'),
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor = "rgba(0,0,0,0)",
                    margin_autoexpand=True,
                    autosize=True,
                    
                )
                
                #fig.show(scale=200, config= dict(displayModeBar = False))
                
                ############################################################
                
                # Writing and exporting interactive figure as html file 
                model_scores_html_path = os.path.join(model_scores_file_dir, "model_score.html")
                f1.write_html(model_scores_html_path, full_html=False, config= dict(displayModeBar = False))

            multi_plot(df1,df2, title="Model Scores")


            #####################################################################

            #model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
            #                                              trained_model_file_path=trained_model_file_path,
            #                                              train_rmse=metric_info.train_rmse,
            #                                              test_rmse=metric_info.test_rmse,
            #                                              train_accuracy=metric_info.train_accuracy,
            #                                              test_accuracy=metric_info.test_accuracy,
            #                                              model_accuracy=metric_info.model_accuracy)


            print('======== Model Trainer: ClassModel Trainer Artifact ======='*2)
            
            print("================================================================"*2)                        


            model_trainer_artifact=  ClassModelTrainerArtifact(is_trained=True,
                                                               message="Model Trained successfully",
                                                               trained_model_file_path=trained_model_file_path,
                                                               train_f1_weighted=metric_info.train_f1_weighted,
                                                               test_f1_weighted=metric_info.test_f1_weighted,
                                                               train_balanced_accuracy=metric_info.train_balanced_accuracy,
                                                               test_balanced_accuracy=metric_info.test_balanced_accuracy,
                                                               model_accuracy=metric_info.model_accuracy)            
       
            print(model_trainer_artifact)
            print("================================================================"*2)                        

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        
            return model_trainer_artifact
        
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")