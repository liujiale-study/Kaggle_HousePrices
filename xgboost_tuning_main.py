# Credits: https://www.kaggle.com/code/merrickolivier/hyperopt-and-xgbregressor-bayesian-optimization
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope

# ==== Configs ====

# Files and Folders
PREPROCESSED_TRAIN_DATA = "Preprocessed_Data/pp_train.csv"


# Validation
NUM_FOLD_CROSS_VALIDATION = 10      # No. of folds for k-fold cross validation
VALIDATION_SET_SPLIT = 0.2          # Percentage of training set to be used as validation set
TRAIN_VALID_SPLIT_RANDOMSEED = 0    # Random seed use for train-validation set split


if __name__ == "__main__":

    # Load data
    train_data = pd.read_csv(PREPROCESSED_TRAIN_DATA)
    y_train_full = train_data.SalePrice
    x_train_full = train_data.drop(columns=["Id","SalePrice"])

    # Partition training and validation sets
    x_train, x_valida, y_train, y_valida = train_test_split(x_train_full, y_train_full, test_size = VALIDATION_SET_SPLIT, random_state = TRAIN_VALID_SPLIT_RANDOMSEED)


    #Define the space over which hyperopt will search for optimal hyperparameters.
    space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 5, 1)),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0,1),
        'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
        'n_estimators': 1000,
        'learning_rate': hp.uniform('learning_rate', 0, .15)}


    # Define the hyperopt objective.
    def func_objective(space):
        model = xgb.XGBRegressor(**space)
        
        # Define evaluation datasets.
        evaluation = [(x_train, y_train), (x_valida, y_valida)]
        
        # Fit the model. Define evaluation sets, early_stopping_rounds, and eval_metric.
        model.fit(x_train, y_train,
                eval_set=evaluation, eval_metric="rmse",
                early_stopping_rounds=100,verbose=False)

        # Obtain prediction and rmse score.
        pred = model.predict(x_valida)
        rmse = mean_squared_error(y_valida, pred, squared=False)
        print ("SCORE:", rmse)
        
        # Specify what the loss is for each model.
        return {'loss':rmse, 'status': STATUS_OK, 'model': model}
    
    trials = Trials()

    best_hyperparams = fmin(fn = func_objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 30,
                            trials = trials)

    print("The best hyperparameters are : ","\n")
    print(best_hyperparams)

    #Create instace of best model.
    best_model = trials.results[np.argmin([r['loss'] for r in 
        trials.results])]['model']