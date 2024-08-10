#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 07:36:04 2024
@author: kaalvoetranger@gmail.com

This script processes multiple CSV files located in specific folders
within a given directory, merges the data from these CSV files into a single
DataFrame, and then saves the merged DataFrame to a CSV

Reports can be downloaded here:
    https://www.eveonline.com/news/search?q=monthly%20economic%20report
    
code block 1 summary:
    defines merge_MER_data function that -
        takes file_name and relative_path as inputs
        iterates through folders in root location (with light error handling)
        merges monthly datasets together and outputs a single 
        merged dataset to root location.
        
code block 2 summary:
   creates a list of files names - 
       runs each filename through the function and also output global variable         
"""
# %% 1. import dependencies and define function

import os
import pandas as pd
import time


def merge_MER_data(file_name, relative_path):
    # Get the absolute path to the specified relative path
    data_root = os.path.expanduser(f'~/{relative_path}')

    # Initialize an empty list to hold all DataFrames
    data_frames = []

    # Iterate over each month's folder
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)

        if os.path.isdir(folder_path):
            # Extract the MonYY from the folder name (e.g., Jan18)
            month_year = folder_name.split('_')[-1]  # Extract MonYY part from folder name

            # Convert the MonYY to a datetime object
            try:
                # Parse the month and year from the folder name using a flexible approach
                date_obj = pd.to_datetime(month_year, errors='coerce')

                if not pd.isnull(date_obj):  # Ensure a valid datetime object
                    # Construct path to file_name within the current month's folder
                    mining_by_region_path = os.path.join(folder_path, file_name)

                    # Read file_name into a DataFrame
                    if os.path.exists(mining_by_region_path):
                        df = pd.read_csv(mining_by_region_path)

                        # Add a new column for history_date
                        df['history_date'] = date_obj

                        # Append the DataFrame to the list
                        data_frames.append(df)

            except ValueError:
                # Handle any parsing errors gracefully
                print(f"Skipping invalid folder name: {folder_name}")

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(data_frames, ignore_index=True)

    # Set the MonthYear column as the index
    merged_df.set_index('history_date', inplace=True)

    # Sort the DataFrame by the index (ascending order of dates)
    merged_df.sort_index(inplace=True)
    print("Datasets have been merged successfully:")
    print(merged_df.info()) ; time.sleep(1)

    # Check which months are included in the merged DataFrame
    included_months = merged_df.index.strftime('%b %Y').unique()
    included_months_count = len(merged_df.index.strftime('%b %Y').unique())
    print(f"Months included in the dataset:\n{', '.join(included_months)}") ; time.sleep(1)
    print(f"Total of {included_months_count} Months appended to output dataset")

    # Save the merged DataFrame locally
    merged_df.to_csv(os.path.join(data_root, f"df_{file_name}"), index=True)
    print(f"df_{file_name.replace('.csv', '')} saved to root path as df_{file_name}")

    # Return the merged DataFrame
    return merged_df




# %% 2. specify file_names and call function

# Specify the relative path to the 'Eve Online MER dump' folder on the desktop
relative_path = 'Desktop/Eve Online MER dump'

# List of file names to process
file_list = ['mining_by_region.csv', 'wormhole-trade.csv', 'regional_stats.csv']

# Loop through each file name and call merge_MER_data function
for file_name in file_list:
    # Call the function and assign the returned DataFrame to a global variable with prefix 'df_'
    global_var_name = f"df_{file_name.replace('.csv', '')}"
    globals()[global_var_name] = merge_MER_data(file_name, relative_path)

    # Print information or perform other actions if needed
    print(f"Processed {file_name} and assigned to global variable {global_var_name}")
    
print("Program Complete")    