# Kaggle_HousePrices
Project code for Kaggle Competition: House Prices - Advanced Regression Techniques [\[Competition Link\]](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)


Folders:
* ```Data```: Contains data files from Kaggle
* ```Preprocessed Data```: Contains CSV files with data that have already been preprocessed.
* ```Solution```: Contains files with prediction results on the test set. Data in solution file is in a format that is accepted by Kaggle.
<br>

Code File Summary:
* ```preprocess_main.py```: Script for processing training and test data from Kaggle. Outputs preprocessed CSV data files in the ```Preprocessed Data``` folder.
* ```model_main.py```: Script with model code. Outputs ```Solution\solution.csv``` that can be submitted to Kaggle.
* ```helper.py```: Script with various utility functions that can be used in any other code files.
<br>

Credits: <br>
Below are codes we referenced when creating our solution.
* ElasticNet and its Hyperparameter Tuning: <https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset>
* Data Preprocessing and Creating Solution from 2 Models: <https://www.kaggle.com/code/apapiu/regularized-linear-models>
* XGBoost Hyperparameter Tuning: <https://www.kaggle.com/code/merrickolivier/hyperopt-and-xgbregressor-bayesian-optimization>
