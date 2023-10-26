# Step 5 (Final): Generate Predictions
# Use best hyperparamters and weight from Step 2 to 4
# Train XGBoost and ElasticNet on the full training dataset
# Create final prediction and CSV for Kaggle submission

from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import xgboost as xgb
import helper

# ==== Model Configs ====
# ==== FILL IN THIS SECTION WITH RESULTS FROM STEP 2 to 4 ====

# ElasticNet Best Params
ELASTICNET_BEST_ALPHA = 0.0006  # Note: Took best validation set result
ELASTICNET_BEST_L1 = 0.735

# XGBoost Best Params
XGB_N_ESTIMATOR = 1000
XGB_BEST_MAX_DEPTH = 5
XGB_BEST_COLSAMPLE = 0.430363
XGB_BEST_MIN_CHILD_W = 0.757017
XGB_BEST_LEARN_RATE = 0.021123

# Best Weight for Combining Model Results
MODEL_WEIGHT_XGB = 0.4  # Final Predictions = weight * XGB Predictions + (1-weight) * ElasticNet Predictions


# ==== Configs ====

# Files and Folders
PREPROCESSED_TRAIN_DATA = "Preprocessed_Data/pp_train.csv"
PREPROCESSED_TEST_DATA = "Preprocessed_Data/pp_test.csv"
OUTPUT_DIR = "Solution"
OUTPUT_FILENAME = "solution.csv"

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
    elasticNet = ElasticNet(alpha = ELASTICNET_BEST_ALPHA, l1_ratio = ELASTICNET_BEST_L1)
    elasticNet.fit(x_train, y_train)

    # XGBoost Setup and Fit
    model_xgb = xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATOR, max_depth=XGB_BEST_MAX_DEPTH, learning_rate=XGB_BEST_LEARN_RATE,
                                 colsample_bytree=XGB_BEST_COLSAMPLE, min_child_weight=XGB_BEST_MIN_CHILD_W)
    model_xgb.fit(x_train, y_train)

    # Generate Predictions
    # Note: Since data pre-processing applied log1p to training targets,
    #       apply expm1 to invert the effects of log1p to get predictions in dollars
    xgb_preds = np.expm1(model_xgb.predict(x_test))
    elasticNet_preds = np.expm1(elasticNet.predict(x_test))
    preds = (1-MODEL_WEIGHT_XGB)*elasticNet_preds + MODEL_WEIGHT_XGB*xgb_preds

    # Setup solution dataframe
    solution = pd.DataFrame({"Id":ids_test, "SalePrice": preds})
    
    # Ensure output folder exist and output to CSV
    helper.validate_output_folder(OUTPUT_DIR)
    helper.output_dataframe_to_folder(solution, OUTPUT_DIR, OUTPUT_FILENAME)


