import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from utils import *
from constants import *
import argparse
import logging
import os


def main(model_name,logger):

    '''
    main function - starting point of the code
    input: model name taken as input 
    (default: adaboost model)
    output: no return value, only prints outcome
    '''
    try:
        print("Starting execution of the inference code...")
        logger.info("Started execution. Fetching data now ...")
        #getting inference data and labels
        inference_data,labels = get_inference_data()
        logger.info("Data fetched. Applying pre-processing now ...")
        #applying preprocessing one_hot_encoding-->normalization
        processed_inference_data = apply_pre_processing(inference_data)
        logger.info("Pre-processing is completed. Loading trained model now ...")
        #loading the model
        model = joblib.load(model_name)
        logger.info("Trained model is loaded. Executing trained model on inference data ...")
        #predecting the output based on the model provided
        model.predict(processed_inference_data)
        #checking the model accuracy
        print("Checking inference accuracy:")
        print(accuracy_score(labels,model.predict(processed_inference_data)))
        logger.info("Execution is complete.")
    
    except Exception as e:
        print("-----!!!ERROR!!!-----")
        logger.error("Encountered error. Please check.")
        logger.error(e)
        print(e)


if __name__== "__main__":
    #creating logging file
    logging.basicConfig(filename="inference_pipe_exec.log",
                        format="%(asctime)s%(message)s",
                        filemode="a")
    
    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    
    parser = argparse.ArgumentParser(description="Running inference pipeline")
    parser.add_argument("--model",
                        default="adaboost",
                        help="select algorithm: svm or adaboost")
    args = parser.parse_args()
    print(f"Selected Model is {args.model}")
    if(args.model == 'svm'):
        model_name = "models/axith_model1_SVM.joblib"
    else:
        model_name = "models/axith_model1_adaboost.joblib"
    main(model_name,logger)