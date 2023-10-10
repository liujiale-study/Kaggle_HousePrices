# Credits: https://www.kaggle.com/code/apapiu/regularized-linear-models
#          https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset
from sklearn.linear_model import ElasticNetCV
import pandas as pd
import numpy as np
import xgboost as xgb
import helper

# Configs
PREPROCESSED_TRAIN_DATA = "Preprocessed_Data/pp_train.csv"
PREPROCESSED_TEST_DATA = "Preprocessed_Data/pp_test.csv"
OUTPUT_DIR = "Solution"
OUTPUT_FILENAME = "solution.csv"

if __name__ == "__main__":

    # Load data
    train_data = pd.read_csv(PREPROCESSED_TRAIN_DATA)
    y_train = train_data.SalePrice
    x_train = train_data.drop(columns=["Id","SalePrice"])

    test_data = pd.read_csv(PREPROCESSED_TEST_DATA)
    ids_test = test_data.Id
    x_test = test_data.drop(columns=["Id"])


    # Setup elasticNet Model
    elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                            alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                        0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                            max_iter = 50000, cv = 10)
    elasticNet.fit(x_train, y_train)

    # XGBoost
    dtrain = xgb.DMatrix(x_train, label = y_train)
    dtest = xgb.DMatrix(x_test)

    # params = {"max_depth":2, "eta":0.1}
    # model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
    model_xgb.fit(x_train, y_train)

    xgb_preds = np.expm1(model_xgb.predict(x_test))
    elasticNet_preds = np.expm1(elasticNet.predict(x_test))

    predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":elasticNet_preds})
    predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

    preds = 0.7*elasticNet_preds + 0.3*xgb_preds

    solution = pd.DataFrame({"Id":ids_test, "SalePrice": preds})
    
    # Ensure output folder exist
    helper.validate_output_folder(OUTPUT_DIR)
    helper.output_dataframe_to_folder(solution, OUTPUT_DIR, OUTPUT_FILENAME)


