import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("merged_data.csv")

# Group the data into chunks of 25 readings
grouped_data = df.groupby(np.arange(len(df)) // 25)

# Calculate statistical parameters for each group
for group_idx, group_data in grouped_data:
    mean_values = group_data[['signal']].mean()
    median_values = group_data[['signal']].median()
    kurtosis_values = group_data[['signal']].apply(lambda x: x.kurtosis())
    skewness_values = group_data[['signal']].apply(lambda x: x.skew())

    # Update DataFrame with calculated values
    for idx, row in group_data.iterrows():
        df.at[idx, 'signal_mean_diff'] = row['signal'] - mean_values['signal']
        df.at[idx, 'signal_median_diff'] = row['signal'] - median_values['signal']
        df.at[idx, 'signal_kurtosis_diff'] = row['signal'] - kurtosis_values['signal']
        df.at[idx, 'signal_skewness'] = skewness_values['signal']

# Print the modified DataFrame
print(df)


# Specify the desired output CSV file path for the modified dataset
output_file_path = "stat_modified_data.csv"

# Save the modified DataFrame to the CSV file
df.to_csv(output_file_path, index=False)

# Print a message indicating that the file has been saved
print(f"Modified dataset saved to {output_file_path}")a
