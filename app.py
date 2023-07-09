from ast import mod
from email import message
from email.mime import application
from ipaddress import collapse_addresses
import pickle
from re import S
from numpy import dtype
import pandas as pd
from sklearn.metrics import f1_score
from LoanApproval.logger import get_log_dataframe
from flask import Flask, request
import sys

import pip
from LoanApproval.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from LoanApproval.logger import logging
from LoanApproval.exception import LoanApprovalException
import os
import sys
import json
from LoanApproval.config.configuration import Configuration
from LoanApproval.constant import CONFIG_DIR, get_current_time_stamp
from LoanApproval.pipeline.pipeline import Pipeline
from LoanApproval.entity.eligibility_predictor import LoanApprovalPredictor, LoanApprovalData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "LoanApproval"
SAVED_MODELS_DIR_NAME = "saved_models"

MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


LoanApproval_DATA_KEY = "LoanApproval_Data"
LoanApproval_PREDICTION_VALUE_KEY = "LoanApproval_prediction_value"


app = Flask(__name__)



@app.route('/artifact', defaults={'req_path': 'LoanApproval'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):

    logging.info(f"req_path: {req_path}")
    
    os.makedirs(req_path, exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        #print(MODEL_DIR)
        #print(LoanApprovalPredictor(MODEL_DIR).get_latest_model_path())
        latest_model_dir = LoanApprovalPredictor(MODEL_DIR).get_latest_model_path()
        head, _ = os.path.split(latest_model_dir)
        model_score_file_dir = os.path.join(head, "score")
        os.makedirs(model_score_file_dir, exist_ok=True)
        
        # Getting Model Name
        best_model_path = os.path.join(head, "model.pkl")
        model = pickle.load(open(best_model_path,"rb"))
        model_name = str(model).replace("()","")
        ###############################

        print(model_score_file_dir)
        
        model_scores = pd.read_csv(os.path.join(model_score_file_dir,"model_score.csv"))
        model_scores.set_index("models",drop=True, inplace=True)
        print(model_scores.columns)
        
        f1_score_train = round(model_scores.loc[model_name,"f1_weighted_train"], 3)
        f1_score_test = round(model_scores.loc[model_name,"f1_weighted_test"], 3)
        #print(f1_score_train)
        
        roc_auc_ovr_weighted_train = round(model_scores.loc[model_name,"roc_auc_ovr_weighted_train"], 3)
        roc_auc_ovr_weighted_test = round(model_scores.loc[model_name,"roc_auc_ovr_weighted_test"], 3)        

        balanced_accuracy_train = round(model_scores.loc[model_name,"balanced_accuracy_train"], 3)
        balanced_accuracy_test = round(model_scores.loc[model_name,"balanced_accuracy_test"], 3)

        log_loss_train = round(model_scores.loc[model_name,"log_loss_train"], 3)
        log_loss_test = round(model_scores.loc[model_name,"log_loss_test"], 3)
        

        return render_template('index.html', f1_score_train = f1_score_train, f1_score_test= f1_score_test, roc_auc_ovr_weighted_test = roc_auc_ovr_weighted_test, roc_auc_ovr_weighted_train = roc_auc_ovr_weighted_train, balanced_accuracy_test=balanced_accuracy_test, balanced_accuracy_train= balanced_accuracy_train, log_loss_test= log_loss_test, log_loss_train= log_loss_train)
    
    except Exception as e:
        return str(e)


@app.route('/reports', methods=['GET', 'POST'])
def render_eda():
    try:
        return render_template('eda.html', report1 = "ProfileReport_1.html")
    except Exception as e:
        return str(e)

@app.route('/reports_1', methods=['GET', 'POST'])
def render_profile_report_1():
    try:
        return render_template("ProfileReport_1.html")
    except Exception as e:
        raise LoanApprovalException(e, sys) from e
    
@app.route('/reports_2', methods=['GET', 'POST'])
def render_profile_report_2():
    try:
        return render_template("ProfileReport_2.html")
    except Exception as e:
        raise LoanApprovalException(e, sys) from e    
    
@app.route('/reports_3', methods=['GET', 'POST'])
def render_drift_report():
    try:
        return render_template("drift_report.html")
    except Exception as e:
        raise LoanApprovalException(e, sys) from e        

@app.route('/reports_4', methods=['GET', 'POST'])
def render_performance_report():
    try:
        return render_template("PerformanceReport.html")
    except Exception as e:
        raise LoanApprovalException(e, sys) from e        

@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    logging.info(f"req_path: view_experiment_hist")

    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    
    logging.info(f"req_path: train")

    message = ""
    pipeline = Pipeline(config=Configuration(time_stamp=get_current_time_stamp()))

    if not Pipeline.experiment.running_status:
        message = """
                    Training Started....                     
                    """
        pipeline.start()
    else:
        message = """
                    Training is Already in progress...
                    """
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    logging.info(f"req_path: predict")

    message = ""
    context = {
        LoanApproval_DATA_KEY: None,
        LoanApproval_PREDICTION_VALUE_KEY: None
    }

    if request.method == 'POST':
        Gender_Male = int(request.form.get('Gender_Male'))
        Married_Yes = int(request.form.get('Married_Yes'))
        Education_Not_Graduate = int(request.form.get('Education_Not_Graduate'))
        Self_Employed_Yes = int(request.form.get('Self_Employed_Yes'))
        credit_history = int(request.form.get('Credit_History'))
        Property_Area = request.form.get('Property_Area')
        Dependent = request.form.get('Dependent')
        ApplicantIncome = float(request.form.get('ApplicantIncome'))
        CoApplicantIncome = float(request.form.get('CoApplicantIncome'))
        LoanAmount = float(request.form.get('LoanAmmount'))
        LoanAmountTerm = float(request.form.get('LoanAmmountTerm'))

        # Define a dictionary to map property area values to variables
        #property_area_mapping = {'Urban': [1, 0, 0],
        #                         'Semiurban': [0, 1, 0],
        #                         'Rural': [0, 0, 1]
        #                         }

        # Process the extracted values
        #Property_Area_Urban, Property_Area_Semiurban, Property_Area_Rural = property_area_mapping.get(Property_Area, [0, 0, 0])


        #dependent_mapping = {'Dependent_1': [1, 0, 0],
        #                         'Dependent_2': [0, 1, 0],
        #                         'Dependent_3': [0, 0, 1]
        #                         }
        #Dependents_1, Dependents_2, Dependents_3 = dependent_mapping.get(Dependent, [0, 0, 0])

        LoanApproval_Data = LoanApprovalData(gender=Gender_Male,
                                             married= Married_Yes,
                                             education= Education_Not_Graduate,
                                             self_employed= Self_Employed_Yes,
                                             credit_history = credit_history,
                                             property_area= Property_Area,                                           
                                             Dependents = Dependent,                                          
                                             ApplicantIncome = ApplicantIncome,
                                             CoapplicantIncome =  CoApplicantIncome,
                                             LoanAmount = LoanAmount,
                                             Loan_Amount_Term= LoanAmountTerm
                                             )
        
        LoanApproval_Data_df = LoanApproval_Data.get_LoanApproval_input_data_frame()
        LoanApproval_predictor = LoanApprovalPredictor(model_dir=MODEL_DIR)
        LoanApproval_prediction = LoanApproval_predictor.predict(X=LoanApproval_Data_df)
        
        context = {
            LoanApproval_DATA_KEY: LoanApproval_Data.get_LoanApproval_data_as_dict(),
            LoanApproval_PREDICTION_VALUE_KEY: LoanApproval_prediction
        }
        
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    logging.info(f"req_path: predict_bulk")
    
    try:
        if request.method == 'POST':
            file = request.files['file']
            LoanApproval_Data_df = pd.read_csv(file)
            try:
                LoanApproval_Data_df = LoanApproval_Data_df.drop("Loan_ID", axis=1)
               
            except:
                pass
                
            LoanApproval_predictor = LoanApprovalPredictor(model_dir=MODEL_DIR)
            try:
                LoanApproval_prediction = LoanApproval_predictor.bulk_prediction(X=LoanApproval_Data_df)
            except Exception as e:
                raise LoanApprovalException(e, sys) from e

            class_mapping = {0: 'Not Eligible',
                                   1: 'Eligible',
                                   }
            
            LoanApproval_Data_df["Predictions"] = LoanApproval_prediction
            LoanApproval_Data_df["Predictions"] = LoanApproval_Data_df["Predictions"].map(class_mapping)  # Major class mapping and replacing numerical rpresentation of major classes with string values
            
            result = LoanApproval_Data_df.to_html(show_dimensions=True,
                                         justify= "center",
                                         classes=['table', 'table-striped'],
                                         border=0.5,
                                         escape=True)

            message = "PREDICTION RESULT"
            
            return render_template("bulk_prediction.html", result=result, message= message)
        
        message = 'Something Is Wrong!'
        return render_template("bulk_prediction.html", result="", message= message)
    
    except Exception as e:
        raise LoanApprovalException(e, sys) from e


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH,
                            data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)

    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False, max_rows=None)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)