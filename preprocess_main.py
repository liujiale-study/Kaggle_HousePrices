# Credits: https://www.kaggle.com/code/apapiu/regularized-linear-models
import pandas as pd
import numpy as np
from scipy.stats import skew
import helper

# Config
TRAIN_DATA = "Data/train.csv"
OUTPUT_DIR = "Preprocessed_Data"

OUTPUT_TRAIN_FILE = "pp_train.csv"

TEST_DATA = "Data/test.csv"
OUTPUT_TEST_FILE = "pp_test.csv"



    
if __name__ == "__main__":
    
    # Read CSVs
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    
    # Concat all data
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
    
    
    # Log transform the training targets
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # Numeric Feature Processing
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    
    # Log transform skewed numeric features:
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    
    # Filling remaining NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())
    
    # Creating corresponding matrices for train and test
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice
    
    # Ensure output folder exist
    helper.validate_output_folder(OUTPUT_DIR)
    
    # Create and output dataframes
    train_preprocessed = pd.concat([train.Id, X_train, y], axis = 1)
    helper.output_dataframe_to_folder(train_preprocessed, OUTPUT_DIR, OUTPUT_TRAIN_FILE)
           
    test_preprocessed = pd.concat([test.Id, X_test], axis = 1)
    helper.output_dataframe_to_folder(test_preprocessed, OUTPUT_DIR, OUTPUT_TEST_FILE)
    
    
    