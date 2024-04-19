#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:42:16 2024
@author: kaalvoetranger@gmail.com



"""
# %% import dependencies and datasets
import requests
import pandas as pd
import os
import time

# GitHub username and repository name
username = "kaalvoetranger-88"
repository_name = "Eve-Online-MER"

# GitHub API endpoint to list repository contents
url = f"https://api.github.com/repos/{username}/{repository_name}/contents/"

# Make GET request to GitHub API
response = requests.get(url)

# Check if request was successful (status code 200)
if response.status_code == 200:
    # Parse JSON response
    files_info = response.json()

    # Extract filenames of only the CSV files into a list
    filenames = [file_info['name'] for file_info in files_info if file_info['type'] == 'file' and 
                 file_info['name'].endswith('.csv')]

    # Print the list of CSV filenames
    print("CSV Files in GitHub repository:")
    for filename in filenames:
        print(filename)
        time.sleep(0.5)
    print()  # Print empty line for separation

    # Dictionary to store file URLs
    file_urls = {}

    # Generate file URLs based on CSV filenames
    for filename in filenames:
        file_url = f"https://raw.githubusercontent.com/{username}/{repository_name}/main/{filename}"
        file_urls[filename] = file_url

    # Dictionary to store pandas DataFrames
    dataframes = {}

    # Loop through each CSV file URL, download and create DataFrame
    for filename, url in file_urls.items():
        response = requests.get(url)
        if response.status_code == 200:
            # Save the CSV file locally
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"CSV File '{filename}' downloaded and saved locally.")

            # Create DataFrame from the CSV file and store in dataframes dictionary
            df = pd.read_csv(filename)
            dataframes[filename[:-4]] = df
            print(f"CSV File '{filename}' loaded into DataFrame.")
            time.sleep(1)

        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")


    # Optionally, you can delete the local CSV files after loading into DataFrames   
    time.sleep(1)
    print("Deleting local CSV files...")
    for filename in file_urls.keys():
        os.remove(filename)
        
    print("Local CSV files have been deleted")
    time.sleep(1)
        
else:
    print(f"Failed to retrieve repository contents. Status code: {response.status_code}")


# Convert 'history_date' columns to datetime objects
for key, df in dataframes.items():
    # Check if 'history_date' column exists
    if 'history_date' in df.columns:
        # Convert 'history_date' column to datetime format
        df['history_date'] = pd.to_datetime(df['history_date'], errors='coerce')  # Convert to datetime
        print(f'history_date column in dataset {key} converted to pandas datetime')
        time.sleep(0.5)


# %% show information on each repository dataset

# Iterate through each dataset in the 'dataframes' dictionary
for dataset_key, dataset_df in dataframes.items():
    # Display information about the current dataset
    print(f"Information about the dataset '{dataset_key}':") ; time.sleep(0.5)
    print(dataset_df.info())
    print()

    # Display summary statistics of the current dataset
    print(f"Summary statistics of the dataset '{dataset_key}':")
    print(dataset_df.describe())
    print()
    time.sleep(1)

# Clean up references to dataset_df and dataset_key after the last dataset is displayed
#del dataset_df, dataset_key, df   


# %% staging datasets

# Initialize variables to track the most recent date
latest_date = None

print('finding mosts recent month in repository:') ; time.sleep(0.5)
# Iterate through each DataFrame to find the most recent 'history_date'

for key, df in dataframes.items():
    # Check if 'history_date' column exists and is in datetime format
    if 'history_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['history_date']):
        # Find the maximum date in the 'history_date' column
        current_latest_date = df['history_date'].max()
        
        # Update 'latest_date' if the current date is more recent
        if latest_date is None or current_latest_date > latest_date:
            latest_date = current_latest_date

# Check if a latest date was found and format it as 'mmmYY' (e.g., Apr22)
if latest_date is not None:
    suffix = latest_date.strftime('%b%y').lower()  # Format the date as 'mmmYY'
else:
    suffix = 'unknown'  # Default suffix if no valid date was found

# Output suffix for debugging
print(f"Most recent date suffix: {suffix}") ; time.sleep(1)


# Output Fact and Dimension Tables

for key, df in dataframes.items():
    # Check if the first column is 'history_date' to determine if it's a fact table
    if df.columns[0] == 'history_date':    
        # Output as 'fact_[key]'
        globals()[f'fact_{key}'] = df
        print(f"DataFrame '{key}' identified as a fact table. Output as 'fact_{key}'.")
        time.sleep(1)
    
    # Check if the first column is 'type_id' or 'dungeon_id' to determine if it's a dimension table
    elif df.columns[0] == 'type_id' or df.columns[0] == 'dungeon_id':
        # Output as 'dim_[key]'
        globals()[f'dim_{key}'] = df
        print(f"DataFrame '{key}' identified as a dimension table. Output as 'dim_{key}'.")
        time.sleep(1)
    
    # For datasets that do not match the above conditions, output as fact tables with '_mmmYY' suffix
    else:
        # Output as 'fact_[key]_apr24'
        globals()[f'fact_{key}_{suffix}'] = df
        print(f"DataFrame '{key}' identified as a fact table (for the most recent month). Output as 'fact_{key}_{suffix}'.")
        time.sleep(1)

print("Processing of Datasets completed. Fact and Dimension Tables ready for Staging")
