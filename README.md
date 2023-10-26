# Kaggle_HousePrices
Project code for Kaggle Competition: House Prices - Advanced Regression Techniques [\[Competition Link\]](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)


### Folders
* ```Data```: Contains data files from Kaggle
* ```Preprocessed Data```: Contains CSV files with data that have already been preprocessed.
* ```Solution```: Contains files with prediction results on the test set. Data in solution file is in a format that is accepted by Kaggle.
<br>

### How to Use
#### Procedure for Generating Solution for Submission to Kaggle
Scripts are numbered by the sequence to run them. Follow the following sequence and any additional configuration changes where necessary.
1. ```1_preprocess_data.py```
    - Script for processing training and test data from Kaggle. (Data from Kaggle should be included in the ```Data``` folder.)
    - Outputs preprocessed CSV data files in the ```Preprocessed Data``` folder.
1. ```2_tune_elasticnet.py```
    - Script for finding best hyperparameters for ElasticNet
1. ```3_tune_xgboost.py```
    - Script for finding best hyperparameters for XGBoost
1. ```4_tune_weight_btwn_model.py```
    - <b>Before running this script, remember to fill in the best hyperparameters for each model</b>
    - Final predictions is created by a weighted sum of XGBoost predictions and ElasticNet predictions
    - The weight used for creating this weighted sum is tuned in this script
1. ```5_model_main.py```:
    - <b>Before running this script, remember to fill in the best model hyperparameters and the weight used for creating weighted results</b>
    - Trains ElasticNet and XGBoost on the training data and obtain predictions from each model on the test data
    - Final predictions is created by a weighted sum of XGBoost predictions and ElasticNet predictions
    - Outputs final predictions to ```Solution\solution.csv``` that can be submitted to Kaggle.

#### Utility Script
The following script(s) can be imported into other scripts in this project to utilize the functions within them.
* ```helper.py```
    - Script with functions for file I/O (e.g. output dataframe to CSV) and ML-related calculations (e.g. RMSE calculation)

<br>

### Credits
Below are codes we referenced when creating our solution.
* ElasticNet and its Hyperparameter Tuning: <https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset>
* Data Preprocessing and Creating Solution from 2 Models: <https://www.kaggle.com/code/apapiu/regularized-linear-models>
* XGBoost Hyperparameter Tuning: <https://www.kaggle.com/code/merrickolivier/hyperopt-and-xgbregressor-bayesian-optimization>
