import pandas as pd

# Iterate over the range of patient IDs
for patient_id in range(1, 36):
    # Read the CSV file
    file_path = f"stacked_data_patient_id_{patient_id}.csv"
    df = pd.read_csv(file_path)

    # Add a new column with constant value equal to the patient ID
    df['target_patient'] = patient_id

    # Save the DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

print("New column 'target_patient' added to all CSV files.")
