# Credits: https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import helper

# ==== Configs ====

# Files and Folders
PREPROCESSED_TRAIN_DATA = "Preprocessed_Data/pp_train.csv"

# Validation
NUM_FOLD_CROSS_VALIDATION = 10      # No. of folds for k-fold cross validation
VALIDATION_SET_SPLIT = 0.2          # Percentage of training set to be used as validation set
TRAIN_VALID_SPLIT_RANDOMSEED = 0    # Random seed use for train-validation set split
    
# ==== Main Function of this Script ====
if __name__ == "__main__":

    # Load data
    train_data = pd.read_csv(PREPROCESSED_TRAIN_DATA)
    y_train_full = train_data.SalePrice
    x_train_full = train_data.drop(columns=["Id","SalePrice"])

    

    # Partition training and validation sets
    x_train, x_valida, y_train, y_valida = train_test_split(x_train_full, y_train_full, test_size = VALIDATION_SET_SPLIT, random_state = TRAIN_VALID_SPLIT_RANDOMSEED)


    # Setup ElasticNet
    elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                            alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                        0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                            max_iter = 50000, cv = NUM_FOLD_CROSS_VALIDATION)
    elasticNet.fit(x_train, y_train)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )

    # See performance on validation set
    y_pred = elasticNet.predict(x_valida)
    print("RMSE Score on Validation Set: " + str(helper.calc_rmse_score(y_valida, y_pred)))

    print("Try again for more precision with l1_ratio centered around " + str(ratio))
    elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                            alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                            max_iter = 50000, cv = NUM_FOLD_CROSS_VALIDATION)
    elasticNet.fit(x_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1    
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )

    # See performance on validation set
    y_pred = elasticNet.predict(x_valida)
    print("RMSE Score on Validation Set: " + str(helper.calc_rmse_score(y_valida, y_pred)))

    print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 
        " and alpha centered around " + str(alpha))
    elasticNet = ElasticNetCV(l1_ratio = ratio,
                            alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 
                                        alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
                                        alpha * 1.35, alpha * 1.4], 
                            max_iter = 50000, cv = NUM_FOLD_CROSS_VALIDATION)
    elasticNet.fit(x_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1    
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )

    # See performance on validation set
    y_pred = elasticNet.predict(x_valida)
    print("RMSE Score on Validation Set: " + str(helper.calc_rmse_score(y_valida, y_pred)))

