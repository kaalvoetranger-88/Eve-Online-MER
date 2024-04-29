#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:04:50 2024

@author: kaalvoetranger@gmail.com
"""

# %% 1. import dependencies and datasets

import os
import time
import calendar
import requests

import pandas as pd
import datetime as datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

# ignore specific types of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

    # Dictionary to store file URLs
    file_urls = {}

    # Generate file URLs based on CSV filenames
    for filename in filenames:
        file_url = f"https://raw.githubusercontent.com/{username}/{repository_name}/main/{filename}"
        file_urls[filename] = file_url

    # Dictionary to store pandas DataFrames
    dataframes = {}

    # Set to keep track of processed df_ files
    processed_df_files = set()

    # Loop through each CSV file URL, download and create DataFrame
    for filename, url in file_urls.items():
        response = requests.get(url)
        if response.status_code == 200:
            # Save the CSV file locally
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"CSV File '{filename}' downloaded and saved locally.")

            # Check if filename starts with 'df_' and Process
            if filename.startswith('df_'):
                # Extract corresponding non-prefixed filename
                base_filename = filename[3:] # Remove 'df_' prefix
                if base_filename in filenames:  # Check if corresponding file exists
                    # Create DataFrame from the CSV file
                    df = pd.read_csv(filename)
                    dataframes[base_filename[:-4]] = df  # Remove '.csv' extension from key
                    print(f"CSV File '{filename}' loaded into DataFrame.")
                    # Track processed df_ file
                    processed_df_files.add(filename)
                    time.sleep(0.2)
                else:
                    print(f"No corresponding file found for '{filename}'. Skipping processing.")                
                
            # Process non-prefixed files
            else:
                if f"df_{filename}" not in processed_df_files:  # Check if df_ file was processed
                    df = pd.read_csv(filename)
                    dataframes[filename[:-4]] = df  # Remove '.csv' extension from key
                    print(f"CSV File '{filename}' loaded into DataFrame.")
                    time.sleep(0.2)
                else:
                    print(f"'{filename}' skipped due to corresponding df_ file.")              

            # Optionally, delete the local CSV file after loading into DataFrame
            os.remove(filename)
            print(f"Local CSV file '{filename}' removed from disk")

        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")

    print("All CSV files have been imported into memory and deleted locally.")
    time.sleep(0.2)

else:
    print(f"Failed to retrieve repository contents. Status code: {response.status_code}")

time.sleep(0.5)
print('Importing of datasets complete.') ; time.sleep(0.5) ; print()
print(f'The following {len(dataframes.keys())} datasets have been imported:')
print(f'{dataframes.keys()}') ; time.sleep(0.2) ; print()
print('Each dataset can be accessed by calling dataframes[_KEY_]') ; time.sleep(0.5)
print('IMPORT DEPENDENCIES AND DATASETS SUCCESSFUL.')


# %% 2. preprocessing datasets

# Convert 'history_date' columns to datetime object and set as index

latest_month = None # initialise empty counter for month
latest_year = None  # initialise empty counter for year
# Loop through the dataframes
for key, df in dataframes.items():
    # Check if 'history_date' column exists
    if 'history_date' in df.columns:
        # Convert 'history_date' column to datetime format
        df['history_date'] = pd.to_datetime(df['history_date'], errors='coerce')  
        print(f'history_date column in dataset {key} converted to pandas datetime')
        df.set_index('history_date', inplace=True)
        print(f'history_date column in dataset {key} set as index')
        time.sleep(0.5)
        
        # Calculate latest month and year for filters
        if latest_month is None or latest_year is None:
            latest_date = df.index.max()
            latest_month = latest_date.month  # Get month
            latest_year = latest_date.year  # Get year
            
    else:
        print(f'{key} does not have datetime')
print(f"Latest month in this Repository is {calendar.month_abbr[latest_month]}{latest_year} taken from {key}")        
print('Preprocessing of datasets complete...') ; time.sleep(0.5) ; print()     

# Display summary info for all processed datasets

# Iterate through each dataset in the 'dataframes' dictionary
print('Summary Information on Imported Datasets:')
for key, df in dataframes.items():
    # Display information about the current dataset
    print(f"Information about the dataset '{key}':") ; time.sleep(0.2)
    print(df.info())
    print()

    # Display summary statistics of the current dataset
    print(f"Summary statistics of the dataset '{key}':")
    print(df.describe())
    print()
    time.sleep(0.5)

  
# %% 3. definitions for plotting functions


# Defining Plot Function for Production vs Destruction vs Mining
def plot_pdm(data, window=21):
    """
    Plot a line chart of production vs Desctruction vs Mining.

    Parameters:
    - data (DataFrame): The DataFrame containing the data. 
    - window (int): Smoothing parameter for lines. Default is 21 days
    """
    df = data.copy()

    # Exclude the datetime column (now the index) from the list of columns to plot
    columns_to_plot = df.columns

    # Calculate rolling mean (moving average) for each column
    df_smoothed = df.rolling(window=window).mean()

    # Set up the plot
    plt.figure(figsize=(15, 8))

    # Define colors for each line (you can customize this list of colors)
    colors = sns.color_palette('husl', n_colors=len(columns_to_plot))

    # Plot smoothed lines for all columns on the same axis
    for i, column in enumerate(columns_to_plot):
        # Remove the last 4 characters from the column name to use as legend label
        label = column[:-4] if column.endswith('_isk') else column
        sns.lineplot(x=df_smoothed.index, y=df_smoothed[column], label=label, color=colors[i])

    # Set title, labels, and legend
    plt.title('Production vs Destruction vs Mining', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Value (Trillions)')
    plt.legend(loc='upper center')  # Adjust legend location if needed

    # Customize y-axis ticks to show values in trillions (abbreviated as 1T, 2T, etc.)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e12:.0f}T'))

    # Set y-axis limits between 0 and 6 trillion
    plt.ylim(0, 6e12)

    # Display the plot
    plt.show()

        
# Defining Plot Function for Regional Statistics   
def plot_regional_stats(data, colors='husl', figsize=(24, 16)):
    """
    Plot Regional Statistics for the most recent month.

    Parameters:
    - data (DataFrame): The DataFrame containing the data. 
    - figsize (tuple, optional): Figure size (width, height) in inches. Default is (20, 12).
    """
    # Find the most recent month in the dataset
    latest_month = data.index.get_level_values('history_date').max().month
    latest_year = data.index.get_level_values('history_date').max().year

    # Filter data for the most recent month
    latest_month_data = data[(data.index.get_level_values('history_date').month == latest_month) &
                             (data.index.get_level_values('history_date').year == latest_year)]

    # Extract columns to plot (excluding specific columns)
    columns_to_plot = [col for col in latest_month_data.columns if col not in ['region_id', 'region_name', 'exports_m3', 'imports_m3', 'moon_mining_value']]

    # Define color palette for each column
    column_colors = sns.color_palette(colors, len(columns_to_plot))

    # Calculate number of rows and columns for subplots
    num_cols = len(columns_to_plot)
    num_rows = 1  # We want all plots in one row

    # Plotting the data
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Iterate over each column to create a horizontal bar chart
    for i, column in enumerate(columns_to_plot):
        # Filter data for the current column
        data_to_plot = latest_month_data.reset_index()

        # Determine the subplot to use based on the number of columns
        if num_cols > 1:
            ax = axes[i]  # Use the i-th subplot
        else:
            ax = axes  # Use the only subplot

        # Create horizontal bar plot
        sns.barplot(x=column, y='region_name', data=data_to_plot, ax=ax, orient='h', palette=[column_colors[i]])

        # Set title for the subplot (column name)
        ax.set_title(column)

        # Remove y-axis label for all but the first subplot
        if i > 0:
            ax.set_ylabel('')

        # Remove borders from the subplot
        sns.despine(left=True, bottom=True, ax=ax)
        
        # Add text annotations (values) to each bar
        for bar in ax.patches:
            value = bar.get_width()  # Get the width of the bar (which corresponds to the value)
            formatted_value = f'{value / 1e12:.1f}T'  # Format the value in trillions
            ax.text(value + 0.05 * ax.get_xlim()[1],  # Position the text slightly to the right of the bar
                    bar.get_y() + bar.get_height() / 2,  # Center the text vertically within the bar
                    formatted_value,  # The formatted value as text
                    va='center', ha='left')  # Alignment settings for vertical and horizontal positioning

        # Remove x-axis labels (ticks) from the subplot
        ax.set_xticks([])

    # Adjust layout for the entire figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.axis('off')
    # Show the plot
    plt.show()


# Defining Plot Function for Ore Type Mined over Time 
def plot_ore_over_time(data, values, palette='husl', figsize=(20, 12)):
    """
    Plot a stacked area chart for security band based on ore type.

    Parameters:
    - data (DataFrame): The DataFrame containing the data. 
    - values (str): Name of the column containing the values for the stacked area chart.
    - palette (str or list, optional): Palette name or list of colors for the plot.
    - figsize (tuple, optional): Figure size (width, height) in inches. Default is (20, 12).
    """
    
    # Reset index to make the specified column a regular column for plotting
    df = data.copy()
    df_reset = df.reset_index()
    
    # Pivot the DataFrame for creating a stacked area chart
    pivot_df = df_reset.pivot(index='history_date', columns='security_band', values=values)
    
    # Fill NaN values with 0 for plotting purposes
    pivot_df.fillna(0, inplace=True)
    
    # Create a stacked area plot
    sns.set_palette(palette)
    pivot_df.plot(kind='area', stacked=True, figsize=figsize, alpha=0.9)
    
    # Adding labels and title
    plt.xlabel('Date')  
    plt.ylabel(f'{values.capitalize()} m3')  
    # Replace underscores with spaces in values for the plot title
    values_title = values.replace('_', ' ')
    plt.title(f'{calendar.month_abbr[latest_month]} {latest_year}: {values_title.capitalize()} Over Time')
    
    # Show legend
    plt.legend(title='Security Band')   
    # Show the plot
    plt.show()


# Defining Plot Function for Ore Type Mined by Region  
def plot_mining_by_region(data, volume_mined, volume_wasted, palette='husl'):
    """
    Plot a stacked bar chart of ore mined and wasted by region for the most recent month.

    Parameters:
    - data (DataFrame): The DataFrame containing the data. 
    - volume_mined (str): Name of the column containing volume of mined ore
    - volume_wasted (str): Name of the column containing volume of residue ore
    """
    
    latest_month = data.index.get_level_values('history_date').max().month
    latest_year = data.index.get_level_values('history_date').max().year

    # Filter data for the most recent month
    latest_month_data = data[(data.index.get_level_values('history_date').month == latest_month) &
                             (data.index.get_level_values('history_date').year == latest_year)]

    # Extract region names and asteroid volumes from the DataFrame
    region_names = latest_month_data['region_name']
    volumes_mined = latest_month_data[volume_mined]
    volumes_wasted = latest_month_data[volume_wasted]

    # Sort indices based on mined asteroid volumes in descending order
    sorted_indices = sorted(range(len(volumes_mined)), key=lambda i: volumes_mined[i], reverse=True)

    # Reorder region names and asteroid volumes (mined and wasted) based on sorted indices
    sorted_region_names = [region_names[i] for i in sorted_indices]
    sorted_volumes_mined = [volumes_mined[i] for i in sorted_indices]
    sorted_volumes_wasted = [volumes_wasted[i] for i in sorted_indices]

    # Plotting the sorted bar chart
    sns.set_palette(palette)
    plt.figure(figsize=(12, 7))  # Set the figure size (width, height) in inches

    # Plot bars for mined volumes (blue)
    plt.bar(sorted_region_names, sorted_volumes_mined, color=sns.color_palette()[0], label='Mined')

    # Plot bars for wasted volumes (orange)
    plt.bar(sorted_region_names, sorted_volumes_wasted, color=sns.color_palette()[1], label='Residue')

    plt.xlabel('Region')  # Add label for the x-axis
    plt.ylabel(f'{volume_mined}')  # Add dynamic label for the y-axis
    plt.title(f'{calendar.month_abbr[latest_month]} {latest_year}: {volume_mined.capitalize()} and {volume_wasted.capitalize()} by Region')

    plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines for the y-axis

    # Customize y-axis tick labels to display values in millions (abbreviated as 'M')
    formatter = ticker.FuncFormatter(lambda x, pos: f'{abs(x) / 1000000:.0f}M')  # Convert to millions (1M = 1 million)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()  # Display legend for the plot

    # Adjust x-axis limits to remove empty space on the left and right
    plt.xlim(-0.5, len(sorted_region_names) - 0.5)  # Adjust x-axis limits based on number of bars
    plt.tight_layout()  # Automatically adjust subplot parameters to fit the plot area
    # Show the plot
    plt.show() 
    
    
    
# Defining Plot Functions for Imports, Exports, & Net Exports by Region



# Defining Plot Functions for Money Supply and Isk Velocity




# %% 4. call plotting functions 

plt.style.use('dark_background')
sns.set_palette('husl')  # Choose a bright/husl/Set1 color palette for better contrast

plot_pdm(dataframes['produced_destroyed_mined'], 
         window=21)

plot_regional_stats(dataframes['regional_stats'], colors='rainbow',
                    figsize=(34, 58))  

plot_ore_over_time(dataframes['mining_history_by_security_band'], 
                   'asteroid_volume_mined', 'bright')
plot_mining_by_region(dataframes['mining_by_region'], 
                      'asteroid_volume_mined', 
                      'asteroid_volume_wasted', 
                      'bright') 

plot_ore_over_time(dataframes['mining_history_by_security_band'], 
                   'gas_volume_mined', 'CMRmap')  
plot_mining_by_region(dataframes['mining_by_region'], 
                      'gas_volume_mined',
                      'gas_volume_wasted', 
                      'CMRmap') 

plot_ore_over_time(dataframes['mining_history_by_security_band'], 
                   'ice_volume_mined', 'Spectral') 
plot_mining_by_region(dataframes['mining_by_region'], 
                      'ice_volume_mined', 
                      'ice_volume_wasted', 
                      'Spectral') 

plot_ore_over_time(dataframes['mining_history_by_security_band'], 
                   'moon_volume_mined', 'winter')
plot_mining_by_region(dataframes['mining_by_region'], 
                      'moon_volume_mined', 
                      'moon_volume_wasted', 
                      'winter') 



