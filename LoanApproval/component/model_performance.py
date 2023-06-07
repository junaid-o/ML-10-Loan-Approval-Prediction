import shutil
from statistics import mode
import natsort
from sklearn.metrics import ConfusionMatrixDisplay
from LoanApproval.logger import logging
from LoanApproval.exception import LoanApprovalException
from LoanApproval.entity.config_entity import ModelEvaluationConfig
from LoanApproval.entity.artifact_entity import ClassModelTrainerArtifact, DataIngestionArtifact, \
    DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, BaseDataTransformationArtifact
from LoanApproval.constant import *
import numpy as np
import os
import sys
from LoanApproval.util.util import write_yaml_file, read_yaml_file, load_object,load_data
from LoanApproval.entity.model_factory import evaluate_classification_model
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

class ModelPerformance:

    #def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ClassModelTrainerArtifact):
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_transformation_artifact: BaseDataTransformationArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ClassModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Performance log started.{'<<' * 30} ")
    
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            #self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.data_validation_artifact = data_validation_artifact

            head, _ = os.path.split(self.model_trainer_artifact.trained_model_file_path)
            head, _ = os.path.split(head)

            self.model_performance_dir  = os.path.join(head, "performance")
    
        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        
    def get_trained_model_and_data(self):
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path


            print("========== model Performance v1: trained model file path =============")
            print(trained_model_file_path)
            print("============================================================="*2)

            #import pickle
            #trained_model_object2 = pickle.load(open(trained_model_file_path, "rb"))
            model = load_object(file_path=trained_model_file_path)
            trained_model_object =  model.trained_model_object

            train_file_path = self.data_transformation_artifact.transformed_resampled_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_non_resampled_test_file_path


            print("========== model Performance v2: train and test file path =============")
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

            print("========= Model Performance ==========="*3)
            print("Converting target column into numpy array.")
            print("======================================"*3)

            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")


            print("========= Model Performance ==========="*3)
            print("Dropping target column from the dataframe.")
            print("======================================"*3)


            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")
            return train_dataframe, test_dataframe, train_target_arr, test_target_arr, trained_model_object
        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        
                
    def get_confusion_metrix(self):
        try:

            train_dataframe, test_dataframe, train_target_arr, test_target_arr, trained_model_object = self.get_trained_model_and_data()
            X_test =  test_dataframe
            y_test = test_target_arr

            #################################################
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), squeeze=True)
            fig.suptitle('Confusion Metrix', fontsize=18, fontweight='bold', y=0.98)
            # Uncomment the axes line if increasing column in above line of code
            try:
                # flattening is required only when nrows or ncols is mor than one
                axes = axes.flatten()
            except:
                pass
            i = 0
            ###################################################

            #for name, clf in {'SVM':svm_model, "AdaBoost":AdaBoost_model, "GBDT":GBDT_model,"KNN":knn_model,"LogReg":lr_model,"RF":RF_model}.items():

            model = {"RF": trained_model_object}
            print(trained_model_object)
            
            class_names = ['Not Eligible','Eligible']

            for name, clf in model.items():
                #print("\nFor ",name)
                
                try:
                    ax = axes[i]
                except:
                    ax= axes
                
                confusionMetrix = ConfusionMatrixDisplay.from_estimator(clf, X=X_test, y= y_test, display_labels= class_names, xticks_rotation='vertical', ax=ax, colorbar=False)
                ax.set_title(name)
                
                i +=1

            plt.tight_layout()    

            ####################    EXPORTING CONFUSION METRIX  #############
            #os.makedirs("Results/Results_Classification_resampled/ConfusionMetrix_resampled", exist_ok=True)
            #plt.savefig(f'Results/Results_Classification_resampled/ConfusionMetrix_resampled/ConfusionMetrix_resampled.svg',format='svg',dpi=600)

            os.makedirs(self.model_performance_dir,exist_ok=True)
            plt.savefig(os.path.join(self.model_performance_dir, "confusion_metrix.svg"), format="svg", bbox_inches='tight', dpi=500)

            #######################################################################
            #plt.show()

        except Exception as e:
            raise LoanApprovalException(e, sys) from e


    def get_roc_auc_curve(self):
        try:
            _, X_test, _, y_test, RF_model = self.get_trained_model_and_data()
            
            # Compute the predicted probabilities for each class
            y_score = RF_model.predict_proba(X_test)

            # Compute the ROC curve and ROC AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])


            # Plot the ROC curves for each class
            plt.figure()
            lw = 2
            colors = ['green','red']
            for i, color in zip(range(2), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                        label='ROC curve of class {0} (AUC = {1:0.2f})'
                        ''.format(i, roc_auc[i]))
                
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Multiclass')
            plt.legend(loc="lower right")


            #######################################

            #os.makedirs(f'Results/Results_Classification_resampled/ROC_AUC_OVR_Curve_resampled', exist_ok=True)
            #plt.savefig(f'Results/Results_Classification_resampled/ROC_AUC_OVR_Curve_resampled/ROC_AUC_OVR_Curve_resampled.svg',format='svg',dpi=600)

            roc_auc_curve_file_path = os.path.join(self.model_performance_dir,"roc_auc.svg")
            plt.savefig(roc_auc_curve_file_path, format="svg", bbox_inches='tight', dpi=500)
            ##########################
            #plt.show()

        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        

    def get_learning_curve(self):
        try:
            X_train, _, y_train, _, RF_model = self.get_trained_model_and_data()
            ############## LEARNING CURVE: Subplot Canvas #####################

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), squeeze=True)
            fig.suptitle('Learning Curve', fontsize=16, fontweight='bold', y=1.0)
            
            try:
                # Needed Only when more than 1 columns or rows are need in above line of code
                axes = axes.flatten()
            except:
                pass

            i = 0
            
            #############  Getting Scores For Learning Curve  ###########

            #model = {"AdaBoost":AdaBoost_model, "KNN":knn_model,"RF":RF_model}
            model = {"RF":RF_model}

            for name, clf in model.items():
                print("\nFor ",name)
                #parameters = param_grid[name]
                #gs = grid_search(clf,parameters)

                try:
                    # Needed only when mor than 1 os or columns have been defined as subplots
                    ax = axes[i]
                except:
                    ax = axes

                i +=1
            
                # split dataset into training and test data
                train_sizes, train_scores, test_scores, fit_time, score_time = learning_curve(clf, X_train, y_train,
                                                                        cv=5,
                                                                        verbose=3,
                                                                        random_state= 2023,
                                                                        #scoring=scoring,
                                                                        train_sizes=np.linspace(0.1, 1.0, 5),
                                                                        return_times= True
                                                                    )
                
                ################# LEARNING CURVE PLOT AND EXPORT  ###############

                # calculate mean and standard deviation of training and test scores
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)

                # plot learning curve
                #plt.figure()
                ax.set_title(f'Learning Curve: {name}')
                ax.set_xlabel('Training examples')
                ax.set_ylabel('Score')
                ax.grid()
                ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
                ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
                ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
                ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
                
                ax.legend(loc='best')

            plt.tight_layout()    
                
            ##################   EXPORTING FIGURE    #####################

            #os.makedirs(f"Results/Results_Classification/LearningCurve_resampled", exist_ok=True)
            
            learning_curve_file_path = os.path.join(self.model_performance_dir, "learning_curve.svg")
            plt.savefig(learning_curve_file_path, format='svg',dpi=500)
            
            #plt.show()
            return train_sizes, train_scores, test_scores, fit_time, score_time

        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        

    def get_scalability_performance(self):
        try:
            X_train, _, y_train, _, RF_model = self.get_trained_model_and_data()
            #################################################
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), squeeze=True)
            fig.suptitle('Scalability & Performance', fontsize=18, fontweight='bold', y=1.0)
            axes = axes.flatten()
            i = 0
            ###################################################


            #for name, clf in {'SVM':svm_model, "AdaBoost":AdaBoost_model, "GBDT":GBDT_model,"KNN":knn_model,"LogReg":lr_model,"RF":RF_model}.items():
            model = {"RF":RF_model}


            for name, clf in model.items():
                print("\nFor ",name)
                
                ax = axes[i]
                ax2 = axes[i+1]
                ax3 = axes[i+2]
                i +=3
            
                # split dataset into training and test data
                train_sizes, train_scores, test_scores, fit_time, score_time = learning_curve(clf, X_train, y_train,
                                                                        cv=4,
                                                                        verbose=3,
                                                                        random_state=100,
                                                                        #scoring=scoring,
                                                                        train_sizes=np.linspace(0.1, 1.0, 5),
                                                                    return_times=True)
                
                # calculate mean and standard deviation of training and test scores
                fit_time_mean = np.mean(fit_time, axis=1)
                fit_time_std = np.std(fit_time, axis=1)
                
                score_time_mean = np.mean(score_time, axis=1)
                score_time_std = np.std(score_time, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                

                # plot learning curve
                #plt.figure()
                ax.set_title(f'Scalability For: {name}')
                ax2.set_title(f'Scalability For: {name}')
                ax3.set_title(f'Performance of: {name}')
                
                ax.set_xlabel('Training examples')
                ax2.set_xlabel('Training examples')
                ax3.set_xlabel('Fit Time')
                
                ax.set_ylabel('Fit Time')
                ax2.set_ylabel('Score Time')
                ax3.set_ylabel('Test Score')
                
                ax.grid()
                ax2.grid()
                ax3.grid()
                
                ax.fill_between(train_sizes, fit_time_mean - fit_time_std, fit_time_mean + fit_time_std, alpha=0.1, color='r')
                ax2.fill_between(train_sizes, score_time_mean - score_time_std, score_time_mean + score_time_std, alpha=0.1, color='g')
                ax3.fill_between(fit_time_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='b')
                
                ax.plot(train_sizes, fit_time_mean, 'o-', color='r', label='Fit Time')
                ax2.plot(train_sizes, score_time_mean, 'o-', color='g', label='Score Time')
                ax3.plot(fit_time_mean, test_scores_mean, 'o-', color='b', label='Fit Time vs score')
                
                #ax.legend(loc='best')
                #ax2.legend(loc='best')

            plt.tight_layout()    
                
            #################### EXPORTING COMPLEXITY CURVES: EXPORT ########################

            #os.makedirs(f'Results/Results_Classification_resampled/ScalabilityPerformance_resampled', exist_ok=True)
            #plt.savefig(f'Results/Results_Classification_resampled/ScalabilityPerformance_resampled/ScalabilityPerformance_resampled.svg',format='svg',dpi=600)

            scalability_curves_file_path = os.path.join(self.model_performance_dir,"scalability_performace.svg")
            plt.savefig(scalability_curves_file_path,format='svg',dpi=600)
                
            #plt.show()

        except Exception as e:
            raise LoanApprovalException(e, sys) from e
        
    def get_merged_performance_report(self):
        try:

                # Set the path to the directory containing the sufolders or HTML files
                model_performance_dir = self.model_performance_dir

                # Create a string to hold the HTML code
                html_code = ''

                # Loop through all directories and files in the directory tree

                for root, dirs, files in os.walk(model_performance_dir):
                    files = natsort.natsorted(files)
                    
                    print(files)    
                    
                    for file in files:
                        # Check if the file is an HTML file
                            ###########################################################
                            
                        if file.endswith('.html') or file.endswith('.svg'):
                            #file_list.append(file)
                            # Read the contents of the file
                                
                            with open(os.path.join(root, file), 'r', encoding="utf-8") as f:
                                file_contents = f.read()

                            # Add the contents of the file to the HTML code string
                            html_code += file_contents
                            ##########################################            

                # Write the HTML code to a new file
                with open(os.path.join(self.model_performance_dir, 'PerformanceReport.html'), 'a', encoding="utf-8") as f:
                                
                    f.write(html_code)
                
                ######################### CLEARING ALL FILES Other THAN ONE SPECIFID FILE ###########################
                #shutil.rmtree(dir_path)    # Clear all the files and folder irrespective to that if they contain data or not

                dir_path = self.model_performance_dir
                except_file = 'PerformanceReport.html'

                for file_name in os.listdir(dir_path):
                    if file_name != except_file:
                        os.remove(os.path.join(dir_path, file_name))                

                ####################### COPYING PERFORMANCE REPORT TO TEMPLATE #######################################

                src_dir = self.model_performance_dir                
                dest_dir = os.path.join("templates")

                # Construct the path to the most recent file
                most_recent_file_path = os.path.join(src_dir, "PerformanceReport.html")

                # Copy the most recent file to the destination directory
                shutil.copy(most_recent_file_path, dest_dir)                              

        except Exception as e:
            raise LoanApprovalException(e, sys) from e

        
    def initiate_performance_evaluation(self):
        try:
            self.get_trained_model_and_data()
            self.get_confusion_metrix()
            self.get_roc_auc_curve()
            self.get_learning_curve()
            self.get_scalability_performance()

            return self.get_merged_performance_report()
        except Exception as e:
            raise LoanApprovalException(e, sys) from e