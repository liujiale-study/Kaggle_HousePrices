# Credits: https://www.kaggle.com/code/apapiu/regularized-linear-models
#          https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import xgboost as xgb
import helper

# ==== Configs ====

# Files and Folders
PREPROCESSED_TRAIN_DATA = "Preprocessed_Data/pp_train.csv"
PREPROCESSED_TEST_DATA = "Preprocessed_Data/pp_test.csv"
OUTPUT_DIR = "Solution"
OUTPUT_FILENAME = "solution.csv"

# Model Params
ELASTICNET_BEST_ALPHA = 0.0006  # Note: Took best validation set result
ELASTICNET_BEST_L1 = 0.735

XGB_N_ESTIMATOR = 1000
XGB_BEST_MAX_DEPTH = 5
XGB_BEST_COLSAMPLE = 0.3939708211204558
XGB_BEST_MIN_CHILD_W = 0.8699738826777037
XGB_BEST_LEARN_RATE = 0.018518854082111784

# Weightage Between Models
# TODO: Grid Search a Better Weight
MODEL_WEIGHT_XGB = 0.3 # Final Predictions = weight * XGB Predictions + (1-weight) * ElasticNet Predictions

# ==== Main Function of this Script ====
if __name__ == "__main__":

    # Load data
    train_data = pd.read_csv(PREPROCESSED_TRAIN_DATA)
    y_train = train_data.SalePrice
    x_train = train_data.drop(columns=["Id","SalePrice"])

    test_data = pd.read_csv(PREPROCESSED_TEST_DATA)
    ids_test = test_data.Id
    x_test = test_data.drop(columns=["Id"])


    # ElasticNet Setup and Fit
    alpha = ELASTICNET_BEST_ALPHA
    elasticNet = ElasticNet(alpha = ELASTICNET_BEST_ALPHA, l1_ratio = ELASTICNET_BEST_L1)
    elasticNet.fit(x_train, y_train)

    # XGBoost Setup and Fit
    model_xgb = xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATOR, max_depth=XGB_BEST_MAX_DEPTH, learning_rate=XGB_BEST_LEARN_RATE,
                                 colsample_bytree=XGB_BEST_COLSAMPLE, min_child_weight=XGB_BEST_MIN_CHILD_W)
    model_xgb.fit(x_train, y_train)

    # Predictions
    xgb_preds = np.expm1(model_xgb.predict(x_test))
    elasticNet_preds = np.expm1(elasticNet.predict(x_test))
    preds = (1-MODEL_WEIGHT_XGB)*elasticNet_preds + MODEL_WEIGHT_XGB*xgb_preds

    # Datafram with solution
    solution = pd.DataFrame({"Id":ids_test, "SalePrice": preds})
    
    # Ensure output folder exist and output to CSV
    helper.validate_output_folder(OUTPUT_DIR)
    helper.output_dataframe_to_folder(solution, OUTPUT_DIR, OUTPUT_FILENAME)


