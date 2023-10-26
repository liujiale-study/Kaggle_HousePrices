# Step 3: Fine-tune XGBoost
# Split original training data into train and validation set
# Find best hyperparamters on training and validation set

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
import numpy as np
import helper

# ==== Configs ====

# Files and Folders
PREPROCESSED_TRAIN_DATA = "Preprocessed_Data/pp_train.csv"

# Train-Validation Data Split
VALIDATION_SET_SPLIT = 0.2          # Percentage of training set to be used as validation set
TRAIN_VALID_SPLIT_RANDOMSEED = 0    # Random seed use for train-validation set split

# Hyperopt Config
RSTATE_SEED = 123


# ==== Main Function of this Script ====
if __name__ == "__main__":

    # Load data
    train_data = pd.read_csv(PREPROCESSED_TRAIN_DATA)
    y_train_full = train_data.SalePrice
    x_train_full = train_data.drop(columns=["Id","SalePrice"])

    # Partition training and validation sets
    x_train, x_valida, y_train, y_valida = train_test_split(x_train_full, y_train_full, test_size = VALIDATION_SET_SPLIT, random_state = TRAIN_VALID_SPLIT_RANDOMSEED)


    # Define the space over which hyperopt will search for optimal hyperparameters.
    space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 10, 1)),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0,1),
        'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
        'n_estimators': 1000,
        'learning_rate': hp.uniform('learning_rate', 0, .15),
        'eval_metric':"rmse",
        'early_stopping_rounds':100}


    # Define the hyperopt objective.
    def func_objective(space):
        model = xgb.XGBRegressor(**space)
        
        # Define evaluation datasets.
        evaluation = [(x_valida, y_valida)]
        
        # Fit the model. Define evaluation sets
        model.fit(x_train, y_train, eval_set=evaluation, verbose=False)

        # Obtain prediction and rmse score.
        pred = model.predict(x_valida)
        rmse = helper.calc_rmse_score(y_valida, pred)
        print ("SCORE:", rmse)
        
        # Specify what the loss is for each model.
        return {'loss':rmse, 'status': STATUS_OK, 'model': model}
    
    trials = Trials()

    best_hyperparams = fmin(fn = func_objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 30,
                            trials = trials,
                            rstate = np.random.default_rng(RSTATE_SEED))

    print("The best hyperparameters are : ","\n")
    print(best_hyperparams)