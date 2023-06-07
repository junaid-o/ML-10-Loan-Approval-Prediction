from typing import List
from setuptools import setup, find_packages


# Declearing Variables for setup function
PROJECT_NAME = "Loan Approval Predictor"
VERSION = "0.0.1"
AUTHOR = "Junaid"
DESCRIPTION = """This is Loan Approval Prediction WebApp Compatible With Python=3.10.11"""
#PACKAGES = ["LoanApproval"]   # Name of package folder

REQUIREMENTS_FILE_NAME = "requirements.txt"


def get_requirements_list() ->  List[str]:
    """
    This function returns list of requirements mentioned in 
    requirements.txt file
    return This function is going to return list 
    which contains name of libraries mentioned in requirements.txt file
    """
        
    with open(REQUIREMENTS_FILE_NAME) as requirement_file:
        return requirement_file.readlines()

setup(name= PROJECT_NAME, version=VERSION, author=AUTHOR, packages= find_packages(), install_requires=get_requirements_list())