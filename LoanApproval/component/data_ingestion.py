import shutil
from sqlite3 import Timestamp

from LoanApproval.constant import CURRENT_TIME_STAMP, get_current_time_stamp

from LoanApproval.entity.config_entity import DataIngestionConfig, BaseDataIngestionConfig
from LoanApproval.exception import LoanApprovalException
from LoanApproval.logger import logging
from LoanApproval.entity.artifact_entity import DataIngestionArtifact, BaseDataIngestionArtifact
import os, sys

import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
import plotly.io as pio
import pandas as pd
import os
import natsort

import ydata_profiling
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import matplotlib.pyplot as plt
import boto3
#from LoanApproval.secrets.secret import AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY, AWS_REGION, AWS_BUCKET_NAME, AWS_FOLDER_NAME


class DataIngestion:
    def __init__(self, data_ingestion_config: BaseDataIngestionConfig):
        try:
            logging.info(f"{'='*20} DATA INGESTION LOG STARTED.{'='*20}")

            # self.data_ingestion_config = data_ingestion_config
            self.current_time_stamp = CURRENT_TIME_STAMP
            self.base_data_ingestion_config = data_ingestion_config
            #self.base_dataset_path = r"LoanApproval\dataset_base"
            self.base_dataset_path =os.path.join("LoanApproval","dataset_base")
            self.raw_data_dir = "raw_data"
            self.dataset_s3_bucket_download_dir = os.path.join("LoanApproval", "s3_bucket", self.current_time_stamp)

            self.profiling_dir_part_1 = os.path.join("LoanApproval\\artifact", "Profiling", self.current_time_stamp, "Part_1")
            os.makedirs(self.profiling_dir_part_1, exist_ok=True)

            self.profiling_dir_part_2 = os.path.join("LoanApproval\\artifact","Profiling",self.current_time_stamp,"Part_2")
            os.makedirs(self.profiling_dir_part_2, exist_ok=True)

        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def get_base_data(self):
        try:
            pd.set_option("display.max_columns", None)

            dataset_base = self.base_dataset_path            
            #raw_data_dir = self.base_data_ingestion_config.raw_data_dir
            raw_data_dir_path = os.path.join(dataset_base, self.raw_data_dir)

            print("=== raw_data_dir_path ==" * 20)
            print("\n\n", raw_data_dir_path)
            print("==" * 20)

            # os.makedirs(raw_data_dir, exist_ok=True)

            logging.info(f"raw_data_dir_path: [ {raw_data_dir_path} ]")
            csv_files = []
            
            logging.info(f"{'='*20} READING BASE DATASET {'='*20} \n\n Walking Through All Dirs In [ {raw_data_dir_path} ] for all .data and .test files")

            # Traverse the directory structure recursively
            for root, dirs, files in os.walk(raw_data_dir_path):
                for file in files:
                    # print(files)
                    # Check if the file is a CSV file
                    if file.endswith(".data") or file.endswith(".csv"):
                        file_path = os.path.join(root, file)
                        # print(file_path)

                        # Read the CSV file into a pandas DataFrame
                        df = pd.read_csv(file_path)
                        #df.columns = columns_list
                        
                        
                        # print(file_path, df.columns)
                        csv_files.append(df)
                        

            logging.info(f"Total [ {len(csv_files)} ] files read in all dirs in [ {raw_data_dir_path} ]")

            print("Number of csv files", len(csv_files))

            df_combined = pd.DataFrame()
            for i in range(len(csv_files)):
                # print(len(csv_files[i].columns))
                # df_name = f"df_{i}"
                df_next = csv_files[i]
                df = pd.concat([df_combined, df_next], axis=0)                

            

            
            ###############################################################################################

            logging.info(f"Handling Duplicats.........")
            
            print(df.columns)
            
            df.drop(["Loan_ID"], axis=1, inplace=True)
            df.drop_duplicates(inplace=True)
    



            ################## EXPORT PROCESSED FILE ################
            print("=="*20)
            print(self.base_data_ingestion_config.processed_data_dir)
            print(self.base_data_ingestion_config.cleaned_data_dir)
            print("=="*20)

            #logging.info(f"Exporting Combined and semi-Cleaned Data to path: [{os.path.join(self.base_dataset_path,self.base_data_ingestion_config.processed_data_dir,self.base_data_ingestion_config.cleaned_data_dir)}]")
            logging.info(f"Exporting Combined and semi-Cleaned Data to path: [{os.path.join(self.base_dataset_path,self.base_data_ingestion_config.processed_data_dir,self.base_data_ingestion_config.cleaned_data_dir)}]")

            processed_data_dir = os.path.join(self.base_dataset_path,
                                              self.base_data_ingestion_config.processed_data_dir,
                                              self.base_data_ingestion_config.cleaned_data_dir,
                                              "raw_data_merged")
            
            os.makedirs(processed_data_dir, exist_ok=True)


            processed_data_file_path = os.path.join(processed_data_dir, "df_combined.csv")
            df.to_csv(processed_data_file_path, index=False)

            logging.info("Merged Data Export Done!")
            logging.info(f"Processed data file path: [ {processed_data_file_path} ]")

            print("====== processed_data_file_path====" * 5)
            print(processed_data_file_path)
            print("====================================" * 5)

            return df

        except Exception as e:
            raise LoanApprovalException(e, sys)

    def get_data_transformer_object(self):
        try:
      
            df = self.get_base_data()

            #######################################    MISSING VALUE IMPUATION    ##########################################################
            logging.info(f"Missing Value Imputation and Handling Categorical Variables")

            
            columns_to_impute = ["Gender", "Married", "Dependents", "Self_Employed","Loan_Amount_Term", "Credit_History"]
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

            df["LoanAmount"] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df[["LoanAmount"]])
            df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('float')
            df['Credit_History'] = df['Credit_History'].astype('float')

            ###################### HANDLINNG CATEGOICAL VARIABLES ###########################

            #df_combined_plot = df_combined.copy()

            columns_list = df.columns.to_list()

            #for feature in columns_list:
            #    if len(df_combined[feature].unique()) <= 3:
            #        # print(df_combined[feature].unique() )
            #        value1 = df[feature].unique()[0]
            #        value2 = df[feature].unique()[1]

            #        df_combined[feature] = df_combined[feature].map({f"{value1}": 0, f"{value2}": 1})

            #        print(feature, df_combined[feature].unique())

            #df_combined = pd.get_dummies(data=df_combined, columns=["referral_source"], drop_first=True)
            #df_combined["Class_encoded"] = LabelEncoder().fit_transform(df_combined["Class"])
            
            df = pd.get_dummies(df, drop_first=True)
            return df
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def outliers_handling(self):
        try:
            ############################## OUTLIERS HANDLING ###############################

            df = self.get_data_transformer_object()

            def outliers_fence(col):
                Q1 = df[col].quantile(q=0.25)
                Q3 = df[col].quantile(q=0.75)
                IQR = Q3 - Q1

                lower_fence = Q1 - 1.5*IQR
                upper_fence = Q3 + 1.5*IQR
                return lower_fence, upper_fence

            columns_to_winsorize = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
            for col in columns_to_winsorize:
                lower_fence, upper_fence = outliers_fence(col)
                df[col] = np.where(df[col] < lower_fence, lower_fence, df[col])
                df[col] = np.where(df[col] > upper_fence, upper_fence, df[col])


            logging.info(f"Outliers Handling Done")


            transformed_data_dir = os.path.join( self.base_dataset_path,
                                                  self.base_data_ingestion_config.processed_data_dir,
                                                  self.base_data_ingestion_config.cleaned_data_dir,
                                                  "processed_data",
                                                  "Cleaned_transformed")
            os.makedirs(transformed_data_dir, exist_ok=True)

            transformed_data_file_path = os.path.join(transformed_data_dir, "df_cleaned.csv")
            df.to_csv(transformed_data_file_path, index=False)
            return df

        except Exception as e:
            raise LoanApprovalException(e, sys) from e


    def profiling_report(self):
        def get_missing_value_fig():
            try:
                df_combined = self.get_base_data()

                plt.figure(figsize=(14, 6), layout="tight")

                # plt.subplot(1,2,1)
                # sns.heatmap(df_combined_orig.isnull(), cbar=False, cmap="viridis", yticklabels=False)
                # plt.title('Missing Values Before', fontdict={'fontsize':20},pad=12)

                # plt.subplot(1,2,2)
                sns.heatmap(df_combined.isnull(), cbar=False, cmap="viridis", yticklabels=False)
                plt.title("Revealed Missing Values", fontdict={"fontsize": 20}, pad=12)
                # plt.show()

                missing_value_fig_path = os.path.join(self.profiling_dir_part_1, "1_missing_values.svg")
                plt.savefig(missing_value_fig_path, dpi=300, bbox_inches="tight")

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_outlier_before_fig():
            try:
                df = self.get_base_data()


                # Define the number of rows and columns for the subplot grid
                num_rows = 2
                num_cols = 2

                # Create a subplot grid with the specified number of rows and columns
                fig = sp.make_subplots(rows=num_rows, cols=num_cols,subplot_titles=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])

                # Loop through each column in the dataframe and add a box plot to the subplot grid
                for idx, col_name in enumerate(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']):
                    row_num = (idx // num_cols) + 1
                    col_num = (idx % num_cols) + 1
                    fig.add_trace(px.box(df[col_name]).data[0], row=row_num, col=col_num,)

                    # Set the title of the subplot grid
                    fig.update_layout(height=500,width=1100, title='Before Handling Outliers and After Missing Value Imputation',
                                    paper_bgcolor = "rgba(0,0,0,0)",
                                    plot_bgcolor = "rgba(0,0,0,0)",                  
                                    )
                    fig.update_traces(marker_color='green')
                # Show the plot

                ###############################
                fig.update_yaxes(showline=False,showgrid=False)
                fig.update_xaxes(showline=False,showgrid=False)
                
                #fig.show()
                ##########################################
                ##########################################

                outlier_fig_before_path = os.path.join(self.profiling_dir_part_1, "2_outliers_before.html")
                pio.write_html(fig, file=outlier_fig_before_path, auto_play=False)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_outlier_after_outlier_handling():
            try:
                df = self.outliers_handling()
                # Define the number of rows and columns for the subplot grid
                num_rows = 2
                num_cols = 2

                # Create a subplot grid with the specified number of rows and columns
                fig = sp.make_subplots(rows=num_rows, cols=num_cols,subplot_titles=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])

                # Loop through each column in the dataframe and add a box plot to the subplot grid
                for idx, col_name in enumerate(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']):
                    row_num = (idx // num_cols) + 1
                    col_num = (idx % num_cols) + 1
                    fig.add_trace(px.box(df[col_name]).data[0], row=row_num, col=col_num,)

                    # Set the title of the subplot grid
                    fig.update_layout(height=500,width=1100, title='After Handling Outliers and After Missing Value Imputation',
                                    paper_bgcolor = "rgba(0,0,0,0)",
                                    plot_bgcolor = "rgba(0,0,0,0)",                  
                                    )
                    fig.update_traces(marker_color='green')
                # Show the plot

                ###############################
                fig.update_yaxes(showline=False,showgrid=False)
                fig.update_xaxes(showline=False,showgrid=False)
                #fig.show()
                ##########################################
                ##########################################
                outliers_fig_after_path = os.path.join(
                    self.profiling_dir_part_1, "3_outliers_after.html"
                )
                pio.write_html(fig, file=outliers_fig_after_path, auto_play=False)
            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_class_pecentage_share():
            try:
                df = self.outliers_handling()

                fig = px.pie(df,names='Loan_Status_Y', hole=0.3)

                fig.update_layout(title="Percentage Share of Loan_Status",
                                height=500,
                                width = 500,
                                paper_bgcolor = "rgba(0,0,0,0)",
                                plot_bgcolor = "rgba(0,0,0,0)",

                        #annotations = [dict(text="Class".title(), showarrow=False)],
                        margin_autoexpand=True,

                        legend=dict(yanchor="bottom",
                                                y=-0.3,
                                                xanchor="center",
                                                x=0.5,
                                                orientation='h'),
                        autosize=True)
                #fig.show()


                class_percentage_share_path = os.path.join(self.profiling_dir_part_1, "4_class_share.html")
                pio.write_html(fig, file=class_percentage_share_path, auto_play=False)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_gender_share():
            try:
                df_combined_plot = self.outliers_handling()

                fig = px.histogram(
                    df_combined_plot, x="Gender_Male", color="Gender_Male", histfunc="count"
                )
                fig.update_layout(
                    height=400,
                    width=500,
                    title="Count Plot For Gender",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                # fig.show()

                gender_share_path = os.path.join(
                    self.profiling_dir_part_1, "6_gender_share.html"
                )
                pio.write_html(fig, file=gender_share_path, auto_play=False)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_comparative_impact():
            try:
                df_combined_plot = self.outliers_handling()

                print("=" * 20)
                print(df_combined_plot.columns)

                col1 = ["Gender_Male", "Married_Yes", "Education_Not Graduate"]
                col2 = ["Self_Employed_Yes", "Loan_Status_Y", "Credit_History"]

                # df = df_combined[cols]

                for col in df_combined_plot[col1].columns:
                    for row in df_combined_plot[col2].columns:
                        if col != row:
                            fig = px.pie(
                                df_combined_plot,
                                names="Loan_Status_Y",
                                facet_col=col,
                                facet_row=row,
                                hole=0.3,
                            )

                            fig.update_layout(
                                height=1000,
                                width=1000,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                # annotations = [dict(text="Class".title(), showarrow=False)],
                                margin_autoexpand=True,
                                legend=dict(
                                    yanchor="bottom",
                                    y=-0.5,
                                    xanchor="center",
                                    x=0.5,
                                    orientation="h",
                                ),
                                autosize=True,
                            )
                            # fig.show()
                            # comparative_impact_path = os.path.join(self.profiling_dir,"7_comparative_impact.html")
                            # pio.write_html(fig,file = comparative_impact_path, auto_play=False)

                            relational_separate_fig_dir = os.path.join(
                                self.profiling_dir_part_2, "relational"
                            )
                            os.makedirs(relational_separate_fig_dir, exist_ok=True)

                            relational_separate_fig_path = os.path.join(
                                relational_separate_fig_dir, f"{col}_vs_{row}.html"
                            )

                            pio.write_html(
                                fig,
                                file=relational_separate_fig_path,
                                auto_play=False,
                                full_html=False,
                            )

                ###########################################################

                # Set the path to the directory containing the sufolders or HTML files
                dir_path = relational_separate_fig_dir

                # Create a string to hold the HTML code
                html_code = ""

                # Loop through all directories and files in the directory tree

                for root, dirs, files in os.walk(dir_path):
                    files = natsort.natsorted(files)

                    print(files)

                    for file in files:
                        # Check if the file is an HTML file
                        ###########################################################

                        if file.endswith(".html") or file.endswith(".svg"):
                            # file_list.append(file)
                            # Read the contents of the file

                            with open(
                                os.path.join(root, file), "r", encoding="utf-8"
                            ) as f:
                                file_contents = f.read()

                            # Add the contents of the file to the HTML code string
                            html_code += file_contents
                            ##########################################

                # Write the HTML code to a new file
                with open(
                    os.path.join(self.profiling_dir_part_2, "7_relational_mrged.html"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    # with open('comparative_impact.html', 'a') as f:

                    f.write(html_code)

                shutil.rmtree(dir_path)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_kde_plot():
            try:
                df_combined_plot = self.outliers_handling()

                col1 = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
                

                # generate a sample dataframe
                df = df_combined_plot.drop(col1, axis=1)

                # create figure and axes objects
                fig, axes = plt.subplots(nrows=5, ncols=3, squeeze=True, figsize=(12, 12))

                # flatten the axes array for easy indexing
                axes = axes.flatten()

                # loop through each column and plot the kde on a separate axis
                for i, col in enumerate(df.columns):
                    sns.kdeplot(df[col], ax=axes[i], fill=True)
                    axes[i].set_title(col)

                # remove any unused axes and add a main title
                for i in range(len(df.columns), len(axes)):
                    fig.delaxes(axes[i])
                fig.suptitle("KDE Plot for All Independent Features", fontsize=14)

                # adjust the spacing between the subplots and show the figure
                fig.tight_layout(pad=2)
                # plt.show()

                #############################################
                kde_plot_path = os.path.join(self.profiling_dir_part_1, "8_kde_plot.svg")
                fig.figure.savefig(kde_plot_path, transparent=True, dpi=300)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_yDataprofile():
            try:
                df = self.outliers_handling()
                #df_major_class = self.get_target_by_major_class()
                #df["major_class"] = df_major_class["major_class"]
                yDataprofile = ProfileReport(
                    df=df,
                    explorative=True,
                    infer_dtypes=True,
                    orange_mode=True,
                    dark_mode=True,
                    tsmode=False,
                    plot={"dpi": 200, "image_format": "svg"},
                    title="Profiling Report",
                    progress_bar=True,
                    html={
                        "style": {"full_width": True, "primary_color": "#000000"},
                        "minify": True,
                    },
                    correlations={
                        "pearson": {"calculate": False},
                        "spearman": {"calculate": False},
                        "kendall": {"calculate": False},
                        "phi_k": {"calculate": False},
                    },
                    missing_diagrams={"bar": True, "matrix": True, "heatmap": True},
                    interactions=None,
                )

                yDataprofile_path = os.path.join( self.profiling_dir_part_1, "0_yDataprofile.html")
                yDataprofile.to_file(yDataprofile_path, silent=True)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_profile_report_1():
            try:
                # Set the path to the directory containing the sufolders or HTML files
                dir_path = self.profiling_dir_part_1

                # Create a string to hold the HTML code
                html_code = ""

                # Loop through all directories and files in the directory tree

                for root, dirs, files in os.walk(dir_path):
                    files = natsort.natsorted(files)

                    print(files)

                    for file in files:
                        # Check if the file is an HTML file
                        ###########################################################

                        if file.endswith(".html") or file.endswith(".svg"):
                            # file_list.append(file)
                            # Read the contents of the file

                            with open(
                                os.path.join(root, file), "r", encoding="utf-8"
                            ) as f:
                                file_contents = f.read()

                            # Add the contents of the file to the HTML code string
                            html_code += file_contents
                            ##########################################

                # Write the HTML code to a new file
                with open(
                    os.path.join(self.profiling_dir_part_1, "ProfileReport_1.html"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    # with open('comparative_impact.html', 'a') as f:

                    f.write(html_code)
                ######################### CLEARING ALL FILES Other THAN ONE SPECIFID FILE ###########################
                # shutil.rmtree(dir_path)    # Clear all the files and folder irrespective to that if they contain data or not

                dir_path = self.profiling_dir_part_1
                except_file = "ProfileReport_1.html"

                for file_name in os.listdir(dir_path):
                    if file_name != except_file:
                        os.remove(os.path.join(dir_path, file_name))

                ####################### COPYING PROFILE_REPORT_1 TO TEMPLATE #######################################

                src_dir = os.path.join("LoanApproval\\artifact", "Profiling")
                dest_dir = os.path.join("templates")

                # Get the list of folders in the source directory
                folders = natsort.natsorted(os.listdir(src_dir))

                # Get the most recent folder
                most_recent_folder = folders[-1]

                # Construct the path to the most recent folder
                most_recent_folder_path = os.path.join(
                    src_dir, most_recent_folder, "Part_1"
                )

                # Get the list of files in the most recent folder
                files = natsort.natsorted(os.listdir(most_recent_folder_path))

                # Get the most recent file
                most_recent_file = files[-1]

                # Construct the path to the most recent file
                most_recent_file_path = os.path.join(
                    most_recent_folder_path, most_recent_file
                )

                # Copy the most recent file to the destination directory
                shutil.copy(most_recent_file_path, dest_dir)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        def get_profile_report_2():
            try:
                # Set the path to the directory containing the sufolders or HTML files
                dir_path = self.profiling_dir_part_2

                # Create a string to hold the HTML code
                html_code = ""

                # Loop through all directories and files in the directory tree

                for root, dirs, files in os.walk(dir_path):
                    files = natsort.natsorted(files)

                    print(files)

                    for file in files:
                        # Check if the file is an HTML file
                        ###########################################################

                        if file.endswith(".html") or file.endswith(".svg"):
                            # file_list.append(file)
                            # Read the contents of the file

                            with open(
                                os.path.join(root, file), "r", encoding="utf-8"
                            ) as f:
                                file_contents = f.read()

                            # Add the contents of the file to the HTML code string
                            html_code += file_contents
                            ##########################################

                # Write the HTML code to a new file
                with open(
                    os.path.join(self.profiling_dir_part_2, "ProfileReport_2.html"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    # with open('comparative_impact.html', 'a') as f:

                    f.write(html_code)
                ######################### CLEARING ALL FILES Other THAN ONE SPECIFID FILE ###########################
                # shutil.rmtree(dir_path)    # Clear all the files and folder irrespective to that if they contain data or not

                dir_path = self.profiling_dir_part_2
                except_file = "ProfileReport_2.html"

                for file_name in os.listdir(dir_path):
                    if file_name != except_file:
                        os.remove(os.path.join(dir_path, file_name))

                ####################### COPYING PROFILE_REPORT_2 TO TEMPLATE #######################################

                src_dir = os.path.join("LoanApproval\\artifact", "Profiling")
                dest_dir = os.path.join("templates")

                # Get the list of folders in the source directory
                folders = natsort.natsorted(os.listdir(src_dir))

                # Get the most recent folder
                most_recent_folder = folders[-1]

                # Construct the path to the most recent folder
                most_recent_folder_path = os.path.join(
                    src_dir, most_recent_folder, "Part_2"
                )

                # Get the list of files in the most recent folder
                files = natsort.natsorted(os.listdir(most_recent_folder_path))

                # Get the most recent file
                most_recent_file = files[-1]

                # Construct the path to the most recent file
                most_recent_file_path = os.path.join(
                    most_recent_folder_path, most_recent_file
                )

                # Copy the most recent file to the destination directory
                shutil.copy(most_recent_file_path, dest_dir)

            except Exception as e:
                raise LoanApprovalException(e, sys) from e

        ############ Calling Sub Functions  ############

        get_yDataprofile()
        get_missing_value_fig()
        get_outlier_before_fig()
        get_outlier_after_outlier_handling()
        get_class_pecentage_share()
        
        get_gender_share()
        get_comparative_impact()
        get_kde_plot()
        get_profile_report_1()
        get_profile_report_2()

    def split_data(self):
        try:

            df = self.outliers_handling()

            logging.info(f"Splitting Data into train and test set")

            #################################   DATA SPLIT BEFORE RESAMPLING  #################################################

            X = df.drop(['Loan_Status_Y'], axis=1)
            y = df['Loan_Status_Y']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                shuffle=True,
                                                                stratify=y,
                                                                random_state=2023)

            start_train_set = None
            start_test_set = None
            start_train_set = pd.concat([X_train, y_train], axis=1)
            start_test_set = pd.concat([X_test, y_test], axis=1)

            print(start_train_set.columns)

            split_data_dir = os.path.join(self.base_dataset_path,
                                          self.base_data_ingestion_config.processed_data_dir,
                                          self.base_data_ingestion_config.cleaned_data_dir,
                                          "processed_data",
                                          "split_data")
            
            train_file_dir = os.path.join(split_data_dir, "train_set")
            test_file_dir = os.path.join(split_data_dir, "test_set")

            print("==== train_file_dir ======" * 20)
            print(train_file_dir)
            print("==========================" * 20)

            if start_train_set is not None:
                os.makedirs(train_file_dir, exist_ok=True)

                logging.info(f"Exporting training data to file:[{train_file_dir}]")

                train_file_path = os.path.join(train_file_dir, "train.csv")
                start_train_set.to_csv(train_file_path, index=False)

            if start_test_set is not None:
                os.makedirs(test_file_dir, exist_ok=True)
                logging.info(f"Exporting test data to file:[{test_file_dir}]")
                test_file_path = os.path.join(test_file_dir, "test.csv")

                start_test_set.to_csv(test_file_path, index=False)

            logging.info(f"Data Split Done!")

            #################################   Returning DataIngestionArtifact#################################################
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"DataIngestion Completed Successfully",)

            logging.info(f"Data Ingestion Artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise LoanApprovalException(e, sys)

    def initiate_data_ingestion(self):
        try:
            
            self.get_base_data()
            self.get_data_transformer_object()
            self.outliers_handling()            
            self.profiling_report()

            return self.split_data()
            
        except Exception as e:
            raise LoanApprovalException(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20} Ingestion log completed {'='*20}\n\n")