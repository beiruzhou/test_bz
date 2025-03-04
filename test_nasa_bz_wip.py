# Databricks notebook source
# MAGIC %md
# MAGIC # Data Load Check

# COMMAND ----------

# MAGIC %md
# MAGIC - Since I'm already using Databricks, I opted to analyze the data within the same environment.
# MAGIC - In Databricks, I don't have access to DBFS (Databricks File System), and uploading individual files is only possible via the UI. As a result, there's no need to use pd.read_csv. However, the data is ingested without column names, so they must be manually assigned in a following step.

# COMMAND ----------

# MAGIC %sql
# MAGIC use com_de_alyt_de

# COMMAND ----------

# MAGIC %md
# MAGIC # Import packages

# COMMAND ----------

# import ace_tools as tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# %pip install package_name

# COMMAND ----------

# pip install numpy==1.20.3

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data as pandas Dataframes

# COMMAND ----------

# Generate table names dynamically
table_names = [f"{prefix}_fd_{str(i).zfill(3)}" for prefix in ["test", "train", "rul"] for i in range(1, 5)]

# Dictionary to store Pandas DataFrames in databricks
pandas_dfs = {table: spark.read.table(table).toPandas() for table in table_names}

# COMMAND ----------

for table_name, df in pandas_dfs.items():
    globals()[table_name] = df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modify Dataframes

# COMMAND ----------

# The test and train data was loaded with 2 empty columns and without the column names
# Assign the new column names for the first 25 columns
new_column_names = [
    "unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"
] + [f"sensor_measurement_{i}" for i in range(1, 22)]  # Sensor measurements from 1 to 21 (total 26 columns)

# List of tables to modify
tables_to_modify = [f"{prefix}_fd_{str(i).zfill(3)}" for prefix in ["test", "train"] for i in range(1, 5)]

for table in tables_to_modify:
    df = globals()[table]  # Retrieve the DataFrame variable from global scope
    df = df.iloc[:, :-2]  # Drop the last 2 columns
    df.columns = new_column_names  # Rename the first 25 columns
    globals()[table] = df  # Assign the modified DataFrame back

# Verify changes
test_fd_001.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check column Types

# COMMAND ----------

test_fd_001.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC # Functions

# COMMAND ----------

# MAGIC %md
# MAGIC - If a specific function is expected to be used multiple times throughout the notebook, it has been encapsulated as a reusable function for easy application across different datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## plot_engine_counts

# COMMAND ----------

def plot_engine_counts(df: pd.DataFrame, unit_column: str = 'unit_number'):
    """
    Plots a bar chart showing the count of data points for each engine.

    Parameters:
    pd.DataFrame: The input DataFrame containing engine data.
    unit_column (str): The column name that represents the unit numbers (default is 'unit_number').

    Returns:
    None
    """
    # Calculate count of each engine
    engine_counts = df[unit_column].value_counts().reset_index()
    engine_counts.columns = ['engine', 'count'] # rename the columns 

    # Sort by 'engine' column
    engine_counts = engine_counts.sort_values('engine')

    # Set figure size
    plt.figure(figsize=(21, 8))

    # Create barplot 
    plt.bar(engine_counts['engine'].astype(str), engine_counts['count'])

    # Labels and title
    plt.title('Count of Data Points of each Engine')
    plt.xlabel('Unit Number')
    plt.ylabel('Count')
    plt.xticks(rotation=90)

    # Show plot
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## plot_max_cycle_per_engine

# COMMAND ----------

def plot_max_cycle_per_engine(df: pd.DataFrame, unit_column: str = 'unit_number', cycle_column: str = 'time_in_cycles'):
    """
    Plots a bar chart showing the maximum cycle per engine.

    Parameters:
    pd.DataFrame: The input DataFrame containing engine cycle data.
    unit_column (str): The column name that represents the unit numbers (default is 'unit_number').
    cycle_column (str): The column name that represents cycle count (default is 'time_in_cycles').

    Returns:
    None
    """
    # Get the maximum 'cycle' value for each engine
    max_cycle_per_engine = df.groupby(unit_column)[cycle_column].max().reset_index()

    # Rename columns
    max_cycle_per_engine.columns = ['engine', 'max_cycle']

    # Sort by 'engine' column
    max_cycle_per_engine = max_cycle_per_engine.sort_values('engine')

    # Set figure size
    plt.figure(figsize=(21, 8))

    # Create bar plot
    plt.bar(max_cycle_per_engine['engine'].astype(str), max_cycle_per_engine['max_cycle'])

    # Labels and title
    plt.title('Maximum Cycle per Engine')
    plt.xlabel('Engine Number')
    plt.ylabel('Maximum Cycle')
    plt.xticks(rotation=90)

    # Show plot
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## compare_max_cycle_and_counts

# COMMAND ----------

def compare_max_cycle_and_counts(df: pd.DataFrame, unit_column : str = 'unit_number', cycle_column : str = 'time_in_cycles') -> pd.DataFrame:
    """
    Compares the maximum cycle value per engine with the count of records for each engine to ensure that every engine has fully complete data.

    Parameters:
    pd.DataFrame: The input DataFrame containing engine data.
    unit_column (str): The column name representing unit numbers (default: 'unit_number').
    cycle_column (str): The column name representing cycles (default: 'time_in_cycles').

    Returns:
    pd.DataFrame: A DataFrame with the comparison results and rows where count does not match max_cycle.
    """
    # Get the maximum 'cycle' value for each engine
    max_cycle_per_engine = df.groupby(unit_column)[cycle_column].max().reset_index()

    # Calculate the number of records per engine
    engine_counts = df[unit_column].value_counts().reset_index()
    engine_counts.columns = [unit_column, 'count']

    # Merge two DataFrames for comparison
    comparison_df = pd.merge(max_cycle_per_engine, engine_counts, on=unit_column)

    # Add a column to check if count is equal to max_cycle
    comparison_df['count_equals_max_cycle'] = comparison_df[cycle_column] == comparison_df['count']

    # Find rows where count does not match max_cycle
    false_values = comparison_df[comparison_df['count_equals_max_cycle'] == False]

    # Print results
    print("Rows where count_equals_max_cycle is False:")
    print(false_values)

    return comparison_df

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Data Statistics

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use describe() & .shape

# COMMAND ----------

# Compute describe() for each DataFrame stored in pandas_dfs
for table in tables_to_modify:
    print(table)
    print(globals()[table].describe())

# COMMAND ----------

for table in tables_to_modify:
    print(table)
    print(globals()[table].shape)

# COMMAND ----------

# MAGIC %md
# MAGIC - After reviewing the PDF, I chose to first analyze the train_fd_001 dataset, as it represents the simplest case with a single condition and a default mode

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial Analysis of train_fd_001: The Simplest Case – One Condition, One Default Mode

# COMMAND ----------

# pd.set_option('display.max_rows',500)

# COMMAND ----------

pd.set_option('display.max_columns',504)

# COMMAND ----------

test_fd_001.describe()

# COMMAND ----------

train_fd_001.describe()

# COMMAND ----------

test_fd_001.describe().transpose()

# COMMAND ----------

train_fd_001.describe().transpose()

# COMMAND ----------

# MAGIC %md
# MAGIC First Oberservations: 
# MAGIC - In both the training and test datasets, certain operational settings remained constant. 
# MAGIC - Additionally, several sensor measurements exhibited no variation.
# MAGIC - Furthermore, several other columns displayed only minimal standard deviations.

# COMMAND ----------

# MAGIC %md
# MAGIC - How many data points does each unit_number or engine have?

# COMMAND ----------

test_fd_001.groupby('unit_number').count()

# COMMAND ----------

# MAGIC %md
# MAGIC - Do the number of data points precisely correspond to the time measured in cycles?

# COMMAND ----------

test_fd_001.groupby('unit_number')['time_in_cycles'].count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Number of Data Points of each Engine

# COMMAND ----------

plot_engine_counts(train_fd_001)

# COMMAND ----------

plot_engine_counts(test_fd_001)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Max Value of each Engine

# COMMAND ----------

plot_max_cycle_per_engine(train_fd_001)

# COMMAND ----------

# MAGIC %md
# MAGIC - Observation: There appear to be a few noticeable outliers.

# COMMAND ----------

max_cycle_per_engine = train_fd_001.groupby('unit_number')["time_in_cycles"].max().reset_index()
max_cycle_per_engine.columns = ['engine', 'max_cycle']

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.boxplot(max_cycle_per_engine['max_cycle'])
plt.title('Boxplot of Maximum Cycle per Engine')
plt.ylabel('Maximum Cycle')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Ensure we thoroughly inspect the engines with exceptionally long cycle times and consider excluding these outliers to prevent them from skewing the model.

# COMMAND ----------

plot_max_cycle_per_engine(test_fd_001)

# COMMAND ----------

# To make sure that really every engine has complete data
# Get the maximum 'cycle' value for each engine
max_cycle_per_engine_test = test_fd_001.groupby('unit_number')['time_in_cycles'].max().reset_index()

# Check column names
print(max_cycle_per_engine_test.columns)

# Calculate the number of records per engine
engine_counts = test_fd_001['unit_number'].value_counts().reset_index()
engine_counts.columns = ['unit_number', 'count']

# Check the column names
# print(engine_counts.columns)

# Merge two DataFrames for comparison
comparison_df = pd.merge(max_cycle_per_engine_test, engine_counts, on='unit_number')

# Check the names of the merged DataFrame columns
# print(comparison_df.columns)

# Add a column to check if count is equal to max_cycle
comparison_df['count_equals_max_cycle'] = comparison_df['time_in_cycles'] == comparison_df['count']

# Check for a value of False
false_values = comparison_df[comparison_df['count_equals_max_cycle'] == False]
print("Rows where count_equals_max_cycle is False:")
print(false_values)

# COMMAND ----------



# COMMAND ----------

def compare_max_cycle_and_counts(df: pd.DataFrame, unit_column='unit_number', cycle_column='time_in_cycles'):
"""
    Compares the maximum cycle value per engine with the count of records for each engine.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing engine data.
    unit_column (str): The column name representing unit numbers (default: 'unit_number').
    cycle_column (str): The column name representing cycles (default: 'time_in_cycles').

    Returns:
    pd.DataFrame: A DataFrame with the comparison results and rows where count does not match max_cycle.
    """

# COMMAND ----------

# TO CONTINUE
# To ensure that every engine has fully complete data.
# Get the maximum 'cycle' value for each engine
max_cycle_per_engine_train = train_fd_001.groupby('unit_number')['time_in_cycles'].max().reset_index()

# Check column names
print(max_cycle_per_engine_train.columns)

# Calculate the number of records per engine
engine_counts = train_fd_001['unit_number'].value_counts().reset_index()
engine_counts.columns = ['unit_number', 'count']

# Check the column names
# print(engine_counts.columns)

# Merge two DataFrames for comparison
comparison_df = pd.merge(max_cycle_per_engine_train, engine_counts, on='unit_number')

# Check the names of the merged DataFrame columns
# print(comparison_df.columns)

# Add a column to check if count is equal to max_cycle
comparison_df['count_equals_max_cycle'] = comparison_df['time_in_cycles'] == comparison_df['count']

# Check for a value of False
false_values = comparison_df[comparison_df['count_equals_max_cycle'] == False]
print("Rows where count_equals_max_cycle is False:")
print(false_values)

# COMMAND ----------

# Get the min 'cycle' value for each engine
min_cycle_per_engine = train_fd_001.groupby('unit_number')['time_in_cycles'].min().reset_index()

# Rename columns
min_cycle_per_engine.columns = ['engine', 'min_cycle']

# Sort by 'engine' column
min_cycle_per_engine = min_cycle_per_engine.sort_values('engine')

# Drawing bar graphs
plt.figure(figsize=(21, 8))
plt.bar(min_cycle_per_engine['engine'].astype(str), min_cycle_per_engine['min_cycle'])
plt.title('Minimum Cycle per Engine')
plt.xlabel('Engine Number')
plt.ylabel('Minumum Cycle')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# Get the min 'cycle' value for each engine
min_cycle_per_engine = test_fd_001.groupby('unit_number')['time_in_cycles'].min().reset_index()

# Rename columns
min_cycle_per_engine.columns = ['engine', 'min_cycle']

# Sort by 'engine' column
min_cycle_per_engine = min_cycle_per_engine.sort_values('engine')

# Drawing bar graphs
plt.figure(figsize=(21, 8))
plt.bar(min_cycle_per_engine['engine'].astype(str), min_cycle_per_engine['min_cycle'])
plt.title('Minimum Cycle per Engine')
plt.xlabel('Engine Number')
plt.ylabel('Minumum Cycle')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Integrity: for each cycle a data point?

# COMMAND ----------

# Get the maximum 'cycle' value for each engine
max_cycle_per_engine = train_fd_001.groupby('unit_number')['time_in_cycles'].max().reset_index()

# Check column names
print(max_cycle_per_engine.columns)

# Calculate the number of records per engine
engine_counts = train_fd_001['unit_number'].value_counts().reset_index()
engine_counts.columns = ['unit_number', 'count']

# Check the column names
# print(engine_counts.columns)

# Merge two DataFrames for comparison
comparison_df = pd.merge(max_cycle_per_engine, engine_counts, on='unit_number')

# Check the names of the merged DataFrame columns
# print(comparison_df.columns)

# Add a column to check if count is equal to max_cycle
comparison_df['count_equals_max_cycle'] = comparison_df['time_in_cycles'] == comparison_df['count']

# Check for a value of False
false_values = comparison_df[comparison_df['count_equals_max_cycle'] == False]
print("Rows where count_equals_max_cycle is False:")
print(false_values)

# COMMAND ----------

# Get the maximum 'cycle' value for each engine
max_cycle_per_engine = test_fd_001.groupby('unit_number')['time_in_cycles'].max().reset_index()

# Check column names
print(max_cycle_per_engine.columns)

# Calculate the number of records per engine
engine_counts = test_fd_001['unit_number'].value_counts().reset_index()
engine_counts.columns = ['unit_number', 'count']

# Check the column names
# print(engine_counts.columns)

# Merge two DataFrames for comparison
comparison_df = pd.merge(max_cycle_per_engine, engine_counts, on='unit_number')

# Check the names of the merged DataFrame columns
# print(comparison_df.columns)

# Add a column to check if count is equal to max_cycle
comparison_df['count_equals_max_cycle'] = comparison_df['time_in_cycles'] == comparison_df['count']

# Check for a value of False
false_values = comparison_df[comparison_df['count_equals_max_cycle'] == False]
print("Rows where count_equals_max_cycle is False:")
print(false_values)

# COMMAND ----------

# MAGIC %md
# MAGIC - Data Integrity: The number of records for each engine is consistent with its maximum cycle value, indicating that there is a corresponding record for each cycle in the dataset and no records were missed.

# COMMAND ----------

# MAGIC %md
# MAGIC - Data Quality: No cases were found where the number of records was not equal to the maximum/miminum cycle value, indicating that the dataset was perfect and there were no duplicates or missing records.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Histogram of Time_in_Cycle / Line Counts per Unit

# COMMAND ----------

unit_cycle_counts = train_fd_001.groupby('unit_number')['time_in_cycles'].count()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(unit_cycle_counts, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Cycles per Unit')
plt.ylabel('Frequency')
plt.title('Histogram of Cycle Counts per Unit')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# COMMAND ----------

unit_cycle_counts = test_fd_001.groupby('unit_number')['time_in_cycles'].count()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(unit_cycle_counts, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Cycles per Unit')
plt.ylabel('Frequency')
plt.title('Histogram of Cycle Counts per Unit')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# COMMAND ----------

unit_cycle_counts = train_fd_001.groupby('unit_number')['time_in_cycles'].count()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(unit_cycle_counts, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Cycles per Unit')
plt.ylabel('Frequency')
plt.title('Histogram of Cycle Counts per Unit')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# COMMAND ----------

test_fd_001.unit_number.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Unit Numbers for Engines

# COMMAND ----------

unique_unit_numbers = test_fd_001["unit_number"].unique()
print(unique_unit_numbers)

# COMMAND ----------

# MAGIC %md
# MAGIC - 100 Engines

# COMMAND ----------

# MAGIC %md
# MAGIC - TODO: are these the same 100 engines across all datasets?

# COMMAND ----------

tables_to_modify

# COMMAND ----------

# Dictionary to store unique unit numbers for each DataFrame
unique_unit_numbers = {table: set(globals()[table]["unit_number"].unique()) for table in tables_to_modify}

# Use the first table in the list as the reference
reference_table = tables_to_modify[0]
reference_units = unique_unit_numbers[reference_table]

# Check if all DataFrames have the same unique unit numbers
consistency_check = {table: reference_units == units for table, units in unique_unit_numbers.items()}

# Display results
print(f"Unique Unit Number Check (Reference: {reference_table}):")
for table, is_consistent in consistency_check.items():
    print(f"{table}: {'MATCHES' if is_consistent else 'DOES NOT MATCH'}")

# Optional: Show discrepancies
for table, units in unique_unit_numbers.items():
    if units != reference_units:
        print(f"\nDifferences in {table}:")
        print("Missing in base:", reference_units - units)
        print("Extra in this DF:", units - reference_units)

# COMMAND ----------

# MAGIC %md
# MAGIC - engine numbers: test_fd_001: MATCHES test_fd_003: MATCHES train_fd_001: MATCHES train_fd_003: MATCHES

# COMMAND ----------

# List of dataset pairs to compare
dataset_pairs = [
    ("test_fd_002", "test_fd_004"),
    ("train_fd_002", "train_fd_004")
]

# Loop through each pair and compare unique unit numbers
for dataset1, dataset2 in dataset_pairs:
    unique_units_1 = set(globals()[dataset1]["unit_number"].unique())
    unique_units_2 = set(globals()[dataset2]["unit_number"].unique())

    print(f"\nChecking {dataset1} vs {dataset2}...")
    
    if unique_units_1 == unique_units_2:
        print(f"✅ {dataset1} and {dataset2} have the SAME unique unit numbers.")
    else:
        print(f"❌ {dataset1} and {dataset2} have DIFFERENT unique unit numbers.")
        print(f"Missing in {dataset1}:", unique_units_2 - unique_units_1)
        print(f"Missing in {dataset2}:", unique_units_1 - unique_units_2)

# COMMAND ----------

"""
# List of DataFrames
dataframes = {
    "test_fd_002": test_fd_002,
    "test_fd_003": test_fd_003,
    "test_fd_004": test_fd_004,
    "train_fd_001": train_fd_001,
    "train_fd_002": train_fd_002,
    "train_fd_003": train_fd_003,
    "train_fd_004": train_fd_004
}

# Get unique unit numbers from each DataFrame
unique_unit_numbers = {name: set(df["unit_number"].unique()) for name, df in dataframes.items()}

# Check if all DataFrames have the same unique unit numbers
# base_units = unique_unit_numbers["test_fd_002"]  # Use the first DF as reference
consistency_check = {name: unique_unit_numbers == units for name, units in unique_unit_numbers.items()}

# Display results
print("Unique Unit Number Check Across DataFrames:")
for name, is_consistent in consistency_check.items():
    print(f"{name}: {'MATCHES' if is_consistent else 'DOES NOT MATCH'}")

# Optional: Show any discrepancies
for name, units in unique_unit_numbers.items():
    if units != base_units:
        print(f"\nDifferences in {name}:")
        print("Missing in base:", base_units - units)
        print("Extra in this DF:", units - base_units)
"""

# COMMAND ----------

# MAGIC %md
# MAGIC #### test_fd_002

# COMMAND ----------

test_fd_002.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC - more standard deviation for operational_seeting_1 and 3 as in ds 001, which is expected
# MAGIC - at the same time more deviation for sensor measurements

# COMMAND ----------

train_fd_002.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC - are there any outliers? 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Outliers

# COMMAND ----------

# MAGIC %md
# MAGIC #### Boxplots

# COMMAND ----------

df_plot_train = train_fd_001.drop(columns=['unit_number'])
plt.figure(figsize=(15, 27))
for i, col in enumerate(df_plot_train.columns, start=1):  # without the first column
    temp = train_fd_001[col]  
    plt.subplot(9, 3, i)  
    plt.boxplot(temp)
    plt.title(f"Box Plot of: {col}")  
plt.tight_layout()  # Adjust layout for readability
plt.show()

# COMMAND ----------

df_plot_test = test_fd_001.drop(columns=['unit_number'])

# COMMAND ----------

plt.figure(figsize=(15, 27))
for i, col in enumerate(df_plot_test.columns, start=1):  # without the first column
    temp = test_fd_001[col]  
    plt.subplot(9, 3, i)  
    plt.boxplot(temp)
    plt.title(f"Box Plot of: {col}")  

plt.tight_layout()  # Adjust layout for readability
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Check Sensor 6 / max. Cycle test/train -> yes, can be excluded

# COMMAND ----------

# Function to detect outliers using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Find outliers in sensor_6 for train and test sets
outliers_train = detect_outliers_iqr(train_fd_001, "sensor_measurement_6")
outliers_test = detect_outliers_iqr(test_fd_001, "sensor_measurement_6")

# Display the outliers
print("Outliers in Train Data (sensor_6):")
print(outliers_train)

print("\nOutliers in Test Data (sensor_6):")
print(outliers_test)

# COMMAND ----------

train_fd_001.loc[:,['sensor_measurement_6']]

# COMMAND ----------

train_fd_001[train_fd_001.sensor_measurement_6!= 21.61]

# COMMAND ----------

test_fd_001.loc[:,['sensor_measurement_6']]

# COMMAND ----------

# Plot histograms for sensor_6 in both datasets
plt.figure(figsize=(12, 5))

# Histogram for train data
plt.subplot(1, 2, 1)
sns.histplot(train_fd_001["sensor_measurement_6"], bins=50, kde=True, color="blue")
plt.title("Histogram of Sensor 6 (Train Data)")
plt.xlabel("Sensor 6 Value")
plt.ylabel("Frequency")

# Histogram for test data
plt.subplot(1, 2, 2)
sns.histplot(test_fd_001["sensor_measurement_6"], bins=50, kde=True, color="red")
plt.title("Histogram of Sensor 6 (Test Data)")
plt.xlabel("Sensor 6 Value")
plt.ylabel("Frequency")

# Show the plots
plt.tight_layout()
plt.show()


# COMMAND ----------

max_cycle_per_engine = train_fd_001.groupby('unit_number')['time_in_cycles'].max().reset_index()

# Rename columns
max_cycle_per_engine.columns = ['engine', 'max_cycle']

# COMMAND ----------

# Find outliers in max. cycle for train data set
outliers_train = detect_outliers_iqr(max_cycle_per_engine, "max_cycle")
# outliers_test = detect_outliers_iqr(max_cycle_per_engine, "max_cycle")

# Display the outliers
print("Outliers in Train Data max_cycle:")
print(outliers_train)

# COMMAND ----------

outliers_train

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Remove these 4 Outlier Engines in Max. Time in Cycles in Train Data

# COMMAND ----------

train_fd_001_mod = train_fd_001[~train_fd_001['unit_number'].isin(outliers_train['engine'])]

# COMMAND ----------

train_fd_001.shape

# COMMAND ----------

train_fd_001_mod.shape

# COMMAND ----------

train_fd_001_mod.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Constant Values to Exclude for the Modelling

# COMMAND ----------

# MAGIC %md
# MAGIC - several sensors showing the same value during the period (sensor 1,5,6(one outlier), 10,16,18,19)
# MAGIC - operating setting 3 does not change as expected
# MAGIC - The same oberservation could be made both for train- & test data set 001
# MAGIC - These parameters can be removed from the parameters to be used for modelling to simplify the model

# COMMAND ----------

# List of columns to be deleted
columns_to_drop = [
    "operational_setting_1", "operational_setting_2", "operational_setting_3",
    "sensor_measurement_1", "sensor_measurement_5", "sensor_measurement_6",
    "sensor_measurement_10", "sensor_measurement_16", "sensor_measurement_18",
    "sensor_measurement_19"
]

#train_fd_001_mod2 = train_fd_001_mod.copy()

for col in columns_to_drop:
    if col in train_fd_001_mod.columns:  # Check if the column exists before dropping
        train_fd_001_mod.drop(columns=[col], inplace=True)

# Display the modified DataFrame
display(train_fd_001_mod)  # Databricks function to display DataFrame

# COMMAND ----------

test_fd_001_mod = test_fd_001.copy()

for col in columns_to_drop:
    if col in test_fd_001_mod.columns:  # Check if the column exists before dropping
        test_fd_001_mod.drop(columns=[col], inplace=True)

# Display the modified DataFrame
display(test_fd_001_mod)  # Databricks function to display DataFrame

# COMMAND ----------

train_fd_001_mod.shape

# COMMAND ----------

test_fd_001_mod.shape

# COMMAND ----------

"""
def drop_constant_value(dataframe):
    '''
    Function:
        - Deletes constant value columns in the data set.
        - A constant value is a value that is the same for all data in the data set.
        - A value is considered constant if the minimum (min) and maximum (max) values in the column are the same.
    Args:
        dataframe -> dataset to validate
    Returned value:
        dataframe -> dataset cleared of constant values
    '''

    # Creating a temporary variable to store a column name with a constant value
    constant_column = []

    # The process of finding a constant value by looking at the minimum and maximum values
    for col in dataframe.columns:
        min = dataframe[col].min()
        max = dataframe[col].max()

        # Append the column name if the min and max values are equal.
        if min == max:
            constant_column.append(col)

    # Delete column with constant value
    dataframe.drop(columns=constant_column, inplace=True)

    # return data
    return dataframe

# call function to drop constant value        
data = drop_constant_value(data)
data
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Missing Values

# COMMAND ----------

for table in tables_to_modify:
    print(f"null values in {table}")
    print(globals()[table].isnull().sum())

# COMMAND ----------

for table in tables_to_modify:
    print(f"null values in {table}")
    if globals()[table].isnull().sum().any():
        print(globals()[table].isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC - no missing values

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check duplicates

# COMMAND ----------

test_fd_001.duplicated().sum()

# COMMAND ----------

for table in tables_to_modify:
    print(f"duplicates in {table}")
    if globals()[table].duplicated().sum().any():
        print(globals()[table].duplicated().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ### LATER TODO USE ProfileReport

# COMMAND ----------

# pip install pandas-profiling

# COMMAND ----------

# from pandas_profiling import ProfileReport

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add RUL to Train Data

# COMMAND ----------

df_train_RUL = train_fd_001_mod.groupby(['unit_number']).agg({'time_in_cycles':'max'})

# COMMAND ----------

df_train_RUL

# COMMAND ----------

df_train_RUL.reset_index(inplace=True)

# COMMAND ----------

df_train_RUL.columns

# COMMAND ----------

df_train_RUL.columns = ['unit_number', 'max_cycle']

# COMMAND ----------

train_fd_001_mod = train_fd_001_mod.merge(df_train_RUL,how='left',on=['unit_number'])

# COMMAND ----------

train_fd_001_mod.head()

# COMMAND ----------

train_fd_001_mod['RUL'] = train_fd_001_mod['max_cycle'] - train_fd_001_mod['time_in_cycles']

# COMMAND ----------

train_fd_001_mod.head()

# COMMAND ----------

train_fd_001_mod.drop(['max_cycle'], axis=1, inplace=True)

# COMMAND ----------

train_fd_001_mod.shape

# COMMAND ----------

"""
def add_rul_to_test_data(test_data, test_data_rul):
    """ Enhance each row in the test data with the RUL. This is done inplace.

    :param test_data: The test data to enhance
    :param test_data_rul: The final RUL values for the engines in the test data
    :return: None
    """
    # prepare the RUL file data
    test_data_rul['engine_no'] = test_data_rul.index + 1
    test_data_rul.columns = ['final_rul', 'engine_no']

    # retrieve the max cycles in the test data
    test_rul_max = pd.DataFrame(test_data.groupby('engine_no')['time_in_cycles'].max()).reset_index()
    test_rul_max.columns = ['engine_no', 'max']

    test_data = test_data.merge(test_data_rul, on=['engine_no'], how='left')
    test_data = test_data.merge(test_rul_max, on=['engine_no'], how='left')

    # add the current RUL for every cycle
    test_data['RUL'] = test_data['max'] + test_data['final_rul'] - test_data['time_in_cycles']
    test_data.drop(['max', 'final_rul'], axis=1, inplace=True)

    return test_data
"""

# COMMAND ----------

failure_time = train_fd_001_mod.groupby('unit_number')['RUL'].max()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Histogram

# COMMAND ----------

# distribution of failure time per engine
sns.histplot(failure_time , kde=True)
plt.title('failure time for engine')
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Serie Analysis: Scatter Plot all sensor data

# COMMAND ----------

train_fd_001.columns

# COMMAND ----------

# TRAIN DATA
# Select the range of engine numbers to be plotted
engines_to_plot = range(1, 101)  # Engine numbers from 1 to 100

# Create charts
plt.figure(figsize=(15, 10))

# Iterate over each engine number and plot the lines
for engine_number in engines_to_plot:
    engine_data = train_fd_001[train_fd_001['unit_number'] == engine_number]
    plt.plot(engine_data['time_in_cycles'], engine_data['operational_setting_1'], label=f'Engine {engine_number}')

# Set up chart titles and axis labels
plt.title('Setting1 over Cycles for Engines 1 to 100')
plt.xlabel('Cycle')
plt.ylabel('Setting1 Value')

# To avoid too many legends, only some are shown
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:10], labels[:10], loc='upper right')  # Only the first 10 legends are shown
plt.grid(True)

# COMMAND ----------

# TEST DATA
# Select the range of engine numbers to be plotted
engines_to_plot = range(1, 101)  # Engine numbers from 1 to 100

# Create charts
plt.figure(figsize=(15, 10))

# Iterate over each engine number and plot the lines
for engine_number in engines_to_plot:
    engine_data = test_fd_001[test_fd_001['unit_number'] == engine_number]
    plt.plot(engine_data['time_in_cycles'], engine_data['operational_setting_1'], label=f'Engine {engine_number}')

# Set up chart titles and axis labels
plt.title('Setting1 over Cycles for Engines 1 to 100')
plt.xlabel('Cycle')
plt.ylabel('Setting1 Value')

# To avoid too many legends, only some are shown
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:10], labels[:10], loc='upper right')  # Only the first 10 legends are shown
plt.grid(True)

# COMMAND ----------

# TRAIN DATA
# Select the range of engine numbers to be plotted
engines_to_plot = range(1, 101)  # Engine numbers from 1 to 100

columns_to_plot = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]

# The list contains all the setting and sensor columns
num_columns = 3
num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns  
fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 3), sharex=True)

# Iterate over each engine number and plot the lines
for engine_number in engines_to_plot:
    engine_data = train_fd_001[train_fd_001['unit_number'] == engine_number]
    for ax, column in zip(axes.flatten(), columns_to_plot):
        ax.plot(engine_data['time_in_cycles'], engine_data[column], label=f'Engine {engine_number}' if engine_number == 1 else "")

# Setting up titles, labels and grids for subgraphs
for ax, column in zip(axes.flatten(), columns_to_plot):
    ax.set_title(f'{column} over Cycles')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True)

# Adjust the layout so that subgraphs do not overlap
plt.tight_layout()
plt.show()

# COMMAND ----------

# TEST DATA
# Select the range of engine numbers to be plotted
engines_to_plot = range(1, 101)  # Engine numbers from 1 to 100

columns_to_plot = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]

# The list contains all the setting and sensor columns
num_columns = 3
num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns  
fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 3), sharex=True)

# Iterate over each engine number and plot the lines
for engine_number in engines_to_plot:
    engine_data = test_fd_001[test_fd_001['unit_number'] == engine_number]
    for ax, column in zip(axes.flatten(), columns_to_plot):
        ax.plot(engine_data['time_in_cycles'], engine_data[column], label=f'Engine {engine_number}' if engine_number == 1 else "")

# Setting up titles, labels and grids for subgraphs
for ax, column in zip(axes.flatten(), columns_to_plot):
    ax.set_title(f'{column} over Cycles')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True)

# Adjust the layout so that subgraphs do not overlap
plt.tight_layout()
plt.show()

# COMMAND ----------

# Compute correlation between 'operational_setting_1' and 'time_in_cycles'
correlation = train_fd_001["operational_setting_1"].corr(train_fd_001["time_in_cycles"])

# Display the correlation result
print(f"Correlation between 'operational_setting_1' and 'time_in_cycles': {correlation:.4f}")

# COMMAND ----------

# Compute correlation between 'operational_setting_1' and 'time_in_cycles'
correlation = train_fd_001["operational_setting_2"].corr(train_fd_001["time_in_cycles"])

# Display the correlation result
print(f"Correlation between 'operational_setting_2' and 'time_in_cycles': {correlation:.4f}")

# COMMAND ----------

# Compute correlation between 'operational_setting_1' and 'time_in_cycles'
correlation = test_fd_001["operational_setting_1"].corr(test_fd_001["time_in_cycles"])

# Display the correlation result
print(f"Correlation between 'operational_setting_1' and 'time_in_cycles': {correlation:.4f}")

# COMMAND ----------

# Compute correlation between 'operational_setting_1' and 'time_in_cycles'
correlation = test_fd_001["operational_setting_2"].corr(test_fd_001["time_in_cycles"])

# Display the correlation result
print(f"Correlation between 'operational_setting_2' and 'time_in_cycles': {correlation:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC - Uses Pandas .corr() method to calculate the Pearson correlation coefficient.
# MAGIC Prints the correlation value to understand the relationship:
# MAGIC Close to +1 → Strong positive correlation.
# MAGIC Close to -1 → Strong negative correlation.
# MAGIC Near 0 → No significant correlation.

# COMMAND ----------

# MAGIC %md
# MAGIC - Also the setting1 & setting2 did not show a clear trend with cycle. This indicates that these setting parameters do not change significantly from cycle to cycle, and can be considered to be independent of the cycle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO Summary EDA

# COMMAND ----------

# MAGIC %md
# MAGIC - Remove outliers: Check engines with particularly long cycle times and consider removing these outliers to avoid impacting the model.
# MAGIC - Remove invalid features: Remove setting and sensor data that remain constant throughout all cycles to simplify the model and improve training efficiency.
# MAGIC - Utilize trending features: Most sensor data trends with cycle, and these features can be used in model training to help predict engine health.
# MAGIC - Handle anomalous data: Sensor data that behaves anomalously (e.g., sensor9 and sensor11) is further analyzed to determine its cause and is handled appropriately in the modeling.
# MAGIC - These findings provide a clear direction for subsequent data preprocessing and model building, which can help improve the accuracy and reliability of predictive maintenance models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detect Features which are highly correlated to reduce dimension / heatmap

# COMMAND ----------

# drop all but one of the highly correlated features
cor_matrix = train_fd_001_mod.corr().abs()

# get upper tri
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# find the correlated columns
corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(corr_features)

# drop the highly correlated features
# df_train.drop(corr_features, axis=1, inplace=True)
# df_test.drop(corr_features, axis=1, inplace=True)

# COMMAND ----------

cor_matrix

# COMMAND ----------

upper_tri

# COMMAND ----------

# drop all but one of the highly correlated features
cor_matrix = test_fd_001_mod.corr().abs()

# get upper tri
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# find the correlated columns
corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(corr_features)

# COMMAND ----------

# Compute correlation matrix
correlation_matrix = train_fd_001_mod.corr()

# Plot heatmap
plt.figure(figsize=(21, 15))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# COMMAND ----------

# Compute correlation matrix
correlation_matrix = test_fd_001_mod.corr()

# Plot heatmap
plt.figure(figsize=(21, 15))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Todo Sensor Column Dict

# COMMAND ----------

"""
Sensor_dictionary={}
dict_list = 
{'sm_1': '(Fan inlet temperature) (◦R)',
 'sm_2': '(LPC outlet temperature) (◦R)',
 'sm_3': '(HPC outlet temperature) (◦R)',
 'sm_4': '(LPT outlet temperature) (◦R)',
 'sm_5': '(Fan inlet Pressure) (psia)',
 'sm_6': '(bypass-duct pressure) (psia)',
 'sm_7': '(HPC outlet pressure) (psia)',
 'sm_8': '(Physical fan speed) (rpm)',
 'sm_9': '(Physical core speed) (rpm)',
 'sm_10': '(Engine pressure ratio(P50/P2)',
 'sm_11': '(HPC outlet Static pressure) (psia)',
 'sm_12': '(Ratio of fuel flow to Ps30) (pps/psia)',
 'sm_13': '(Corrected fan speed) (rpm)',
 'sm_14': '(Corrected core speed) (rpm)',
 'sm_15': '(Bypass Ratio) ',
 'sm_16': '(Burner fuel-air ratio)',
 'sm_17': '(Bleed Enthalpy)',
 'sm_18': '(Required fan speed)',
 'sm_19': '(Required fan conversion speed)',
 'sm_20': '(High-pressure turbines Cool air flow)',
 'sm_21': '(Low-pressure turbines Cool air flow)'}
 """

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reduce Dimention? -> let's drop sensor 9

# COMMAND ----------

train_fd_001_mod.columns

# COMMAND ----------

train_fd_001_mod.drop(columns=['sensor_measurement_9'])

# COMMAND ----------

train_fd_001_mod

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO Normalisation Data?

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO Noises? / Histograms Sensors chosen

# COMMAND ----------

train_fd_001_mod.columns

# COMMAND ----------

selected_sensors = ['sensor_measurement_2',
'sensor_measurement_3', 'sensor_measurement_4', 'sensor_measurement_7',
'sensor_measurement_8', 'sensor_measurement_9', 'sensor_measurement_11',
'sensor_measurement_12', 'sensor_measurement_13',
'sensor_measurement_14', 'sensor_measurement_15',
'sensor_measurement_17', 'sensor_measurement_20',
'sensor_measurement_21']

# COMMAND ----------

# Set up subplots: 3 histograms per row
num_sensors = len(selected_sensors)
rows = (num_sensors // 3) + (num_sensors % 3 > 0)  # Calculate required rows

plt.figure(figsize=(15, rows * 4))  # Adjust figure size

for i, sensor in enumerate(selected_sensors, 1):
    plt.subplot(rows, 3, i)  # Create subplots (3 per row)
    sns.histplot(train_fd_001_mod[sensor], bins=50, kde=True, color="blue")  # Histogram with KDE
    plt.title(f"Histogram of {sensor}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

# Adjust layout and show plots
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 1. sensor_measurement_8 and sensor_measurement_13
# MAGIC Highly skewed with an asymmetric distribution.
# MAGIC There is a clear peak with a long tail, suggesting possible sensor drift or different operational modes.
# MAGIC Action: Consider normalization or transformation (e.g., log transformation).
# MAGIC 2. sensor_measurement_9 and sensor_measurement_14
# MAGIC Extreme right-skewness with values clustering on the lower end.
# MAGIC The long tail suggests potential outliers or a specific failure pattern.
# MAGIC Action: Investigate further with box plots or Z-score analysis.
# MAGIC 3. sensor_measurement_17
# MAGIC The multi-peaked distribution (bimodal/multimodal) suggests different engine operating conditions.
# MAGIC Possible sensor noise or external interference.
# MAGIC Action: Check whether these peaks correspond to different failure modes.
# MAGIC 4. sensor_measurement_12 and sensor_measurement_20
# MAGIC Show some level of skewness, but not as severe as the others.
# MAGIC Might benefit from scaling or standardization.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # TODO Train Data / Model

# COMMAND ----------

train_fd_001.shape

# COMMAND ----------

train_fd_001_mod.shape

# COMMAND ----------

test_fd_001_mod.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

print(train_fd_001_mod.columns)

# COMMAND ----------

features = train_fd_001_mod.columns[2:-1]   # select only the sensor measurement columns
features 

# COMMAND ----------

X = train_fd_001_mod[features]
y = train_fd_001_mod['RUL']

# COMMAND ----------

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

# COMMAND ----------

X_train.shape, X_test.shape

# COMMAND ----------

scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)      # fit on only train dataset
X_test = scaler.transform(X_test)            # use same transofrmation , no fit_traansform
# X_val = scaler.transform(X_val)

# COMMAND ----------

"""
X = df[features]
y = df['rul']

X_train , X_test , y_train , y_test  = train_test_split(X, y, test_size=0.2 , random_state=42)
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO real RUL

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO Comparision with real RULs

# COMMAND ----------

# MAGIC %md
# MAGIC