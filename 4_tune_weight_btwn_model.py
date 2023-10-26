# Step 4: Fine-tune Weight for Model Results
# Split original training data into train and validation set
# Final Predictions is calculated as: weight * XGB Predictions + (1-weight) * ElasticNet Predictions
# Find the best weight for the equation above

import xgboost as xgb
from sklearn.linear_model import ElasticNet
import pandas as pd
from sklearn.model_selection import train_test_split
import helper

# ==== Model Configs ====
# ==== FILL IN THIS SECTION WITH RESULTS FROM STEP 2 & 3 ====

# ElasticNet Best Params
ELASTICNET_BEST_ALPHA = 0.0006  # Note: Took best validation set result
ELASTICNET_BEST_L1 = 0.735

# XGBoost Best Params
XGB_N_ESTIMATOR = 1000
XGB_BEST_MAX_DEPTH = 5
XGB_BEST_COLSAMPLE = 0.430363
XGB_BEST_MIN_CHILD_W = 0.757017
XGB_BEST_LEARN_RATE = 0.021123


# ==== Misc Configs ====

# Files and Folders
PREPROCESSED_TRAIN_DATA = "Preprocessed_Data/pp_train.csv"

# Validation
VALIDATION_SET_SPLIT = 0.2           # Percentage of training set to be used as validation set
TRAIN_VALID_SPLIT_RANDOMSEED = 1    # Random seed use for train-validation set split


# ==== Main Function of this Script ====
if __name__ == "__main__":

    # Load data
    train_data = pd.read_csv(PREPROCESSED_TRAIN_DATA)
    y_train_full = train_data.SalePrice
    x_train_full = train_data.drop(columns=["Id","SalePrice"])

    # Partition training and validation sets
    x_train, x_valida, y_train, y_valida = train_test_split(x_train_full, y_train_full, test_size = VALIDATION_SET_SPLIT, random_state = TRAIN_VALID_SPLIT_RANDOMSEED)

    # ElasticNet Setup and Fit
    elasticNet = ElasticNet(alpha = ELASTICNET_BEST_ALPHA, l1_ratio = ELASTICNET_BEST_L1)
    elasticNet.fit(x_train, y_train)

    # XGBoost Setup and Fit
    model_xgb = xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATOR, max_depth=XGB_BEST_MAX_DEPTH, learning_rate=XGB_BEST_LEARN_RATE,
                                 colsample_bytree=XGB_BEST_COLSAMPLE, min_child_weight=XGB_BEST_MIN_CHILD_W)
    model_xgb.fit(x_train, y_train)

    # Predictions
    xgb_preds = model_xgb.predict(x_valida)
    elasticNet_preds = elasticNet.predict(x_valida)

    arr_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    arr_rmse = []
    for weight in arr_weights:
        preds = (1-weight)*elasticNet_preds + weight*xgb_preds
        rmse = helper.calc_rmse_score(y_valida, preds)
        arr_rmse.append(rmse)
        
    index_of_best_score = arr_rmse.index(min(arr_rmse))
    best_weight = arr_weights[index_of_best_score]

    print("Best weight: " + str(best_weight))