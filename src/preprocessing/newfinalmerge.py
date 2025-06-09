import pandas as pd
import glob

# Define a list to store the DataFrames from each CSV
all_data_frames = []

# Use glob to find all CSV files matching the pattern
csv_files = glob.glob("stacked_data_patient_id_*.csv")  # Adjust the pattern if needed

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    data_frame = pd.read_csv(file)

    # Select only the desired columns ("signal" and "target_patient")
    selected_columns = data_frame[["signal", "target_patient"]]

    # Append the selected columns DataFrame to the list
    all_data_frames.append(selected_columns)

# Concatenate all DataFrames vertically
merged_data = pd.concat(all_data_frames, ignore_index=True)

# Save the merged data to a new CSV file
merged_data.to_csv("merged_data.csv", index=False)

print("Data merged and saved to merged_data.csv")
