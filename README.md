# <font color="red"><strong>Prediction of Elgibility For LoanAproval</strong></font>


## **Background:**

Among all industries, insurance domain has the largest use of analytics & data science methods. This data set would provide you enough taste of working on data sets from insurance companies, what challenges are faced, what strategies are used, which variables influence the outcome etc.

## **Problem Statement**
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

## **Dataset:**
<center>
<a href="https://www.kaggle.com/datasets/ninzaami/loan-predication"><button data-md-button>Dataset</button></a>
</center>

## **Tools & Techniques**

*   `Data versioning` using time stamp
*   `Code versioning` using Git
*   `Modular coding` with separate files for data ingestion, transformation, validation, training, evaluation, performance monitoring, model pusher, model configuration, constants, secret keys, artifacts etc
*   `CI / CD Pipeline` using GitHub Actions
*   `S3 Bucket` for storage of dataset.
*   `Docker` for creating container
*   Custome `logger`
*   Custom `Exception Handler`
*   `Package building` using setuptools
*   `Deplyment` tested on **AWS EC2 instance CI/CD** using Github Action Runner


## **Result**

Best performer in both coditions is `RandomForestClassifier`

*   Scores Achieved:
        

        | Metric                | Train | Test  |
        |-----------------------|-------|-------|
        | F1 weighted           | 0.744 | 0.732 |        
        | ROC AUC OVR Weighted  | 0.891 | 0.87 |
        | Balanced Accuracy     | 0.872 | 0.822 |
        | Log loss              | 0.99| 1.027 |
        


## **Evaluation Metrices**
*   F1 weighted score
*   ROC AUC (One vs Rest) Weighted
*   Balanced Accuracy
*   Precision
*   Log loss
*   ConfusionMetrics
*   Learning Curve
*   Complexity and Scalability


## **Approach**

*   Data collection, cleaning, missing value handling, outlier handling, Data Profiling, exploration.

*   Tested Machine Learning algorithms, including `RandomForestClassifier`, `KNeighborsClassifier`, `AdaBoost` and `GradientBoostingClassifir`, `SVC`.

*   Once the training is completd, model is passed through evaluation phase where it has to pass through set of logical conditons. Only the models above the threshold value of evaluation metrics are consider as accepted model and pushed for integration with FlaskApp

```
f1_logic = (train_f1 >= 0.738) and abs(train_f1 - test_f1) <= 0.009
roc_auc_logic = (roc_auc_ovr_weighted_train >= 0.89) and abs(roc_auc_ovr_weighted_train - roc_auc_ovr_weighted_test) <= 0.02
model_accuracy_logic = (train_balanced_accuracy_score >= base_accuracy) and diff_test_train_acc <= 0.04
loss_logic = (loss_train <= 1.013) and abs(loss_train - loss_test) <= 0.04


if f1_logic and roc_auc_logic and model_accuracy_logic and loss_logic:
        -------
        -------
        ------
```

*   Profiling Report, EDA Report and Evaluation Report generation

## **API and Web UI**

*   API exposed via `Flask-Web-App`
*   Dashboard displays score cards for `F1_weighted`, `ROC_AUC_OVR_Weighted`, `Balanced Accuracy`, `Log_loss`
*   Web dashboard allow you:
    *   View all reports for the deployed model:
        *   Profiling Report
        *   EDA Report
        *   Model Performance Report
    
    *   View, modify model configuration and save changes
    *   View and download models accepted above a threshold value of evaluation metrics
    *   Trigger model training
    *   View Logs
    *   View all the artifacts
    *   View history of model training

# **Installation**


## **Requirements**

*   Python 3.10.10
*   Scikit-learn
*   Seaborn
*   Matplotlib
*   Plotly
*   Pandas
*   Numpy
*   Imbalanced Learn
*   PyYAML
*   dill
*   six
*   Flask
*   gunicorn
*   natsort
*   Evidently
*   yData Profiling
*   boto3


## **Docker**

A Dockerfile is a text document that contains all the commands a user could call on the command line to assemble an image. Docker images can be used to create containers, which are isolated environments that run your application. This is useful because it ensures that your application runs in the same environment regardless of where it is being deployed.

To build and run this project using Docker, follow these steps:

1.  Install Docker on your machine if you haven't already.
2.  Open a terminal window and navigate to the project root directory.
3.  Build the Docker image by running the following command:

    ```
    docker build -t <image-name>:<version> <location-of-docker-file for curren directory just add dot (.)>

    ```
    or

    ```
    docker build -t <image-name>:<version> .
    
    ```

4.  To Check List of Docker Images
    ```
    docker images
    ```    

5.  Start a container using the following command, replacing <image-name> and <version> with the values you used in step 3:

    ```
    docker run -p <host-port>:<container-port> <image-name>:<version>

    ```

    or

    ```
    docker run -p 5000:5000 -e PORT=5000 <Image-ID>
    ```

6.  Open a web browser and go to `http://localhost:<host-port>` to see the application running.

7.  Check Running Containers in docker

    ```
    docker ps
    ```

8.  Stop Docker Container

    ```
    docker stop <container_id>
    ```    


# **Project Structure**

