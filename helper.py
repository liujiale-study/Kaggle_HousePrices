import os
from sklearn.metrics import mean_squared_error

# Checks output folder exist and creates it if it doesnt
def validate_output_folder(output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

# Output given dataframe (df) to the given folder
def output_dataframe_to_folder(df, path_to_folder, filename, save_index = False):
    output_path = os.path.join(path_to_folder, filename)
    df.to_csv(output_path, index=save_index)
    
    print("Output file created: " + output_path)

# Calculate root mean squared error
def calc_rmse_score(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared = False)