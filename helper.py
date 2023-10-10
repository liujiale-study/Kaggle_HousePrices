import os
import pandas as pd

# Checks output folder exist and creates it if it doesnt
def validate_output_folder(output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

def output_dataframe_to_folder(df, path_to_folder, filename, save_index = False):
    output_path = os.path.join(path_to_folder, filename)
    df.to_csv(output_path, index=save_index)
    
    print("Output file created: " + output_path)