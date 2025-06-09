import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the PPG data
ppg_data = pd.read_csv('merged_data.csv')
ppg_signal = ppg_data['signal'].values

# Use a smaller subset of the data for visualization
ppg_signal_subset = ppg_signal[50:150]  # Adjust the range as needed

# Detect peaks
peaks, _ = find_peaks(ppg_signal_subset, distance=10)

# Plot the raw PPG signal
plt.figure(figsize=(14, 8))
plt.plot(ppg_signal_subset, label='Raw PPG Signal')
plt.plot(peaks, ppg_signal_subset[peaks], 'ro', label='Peaks')

# Add horizontal lines for peaks
for peak in ppg_signal_subset[peaks]:
    plt.axhline(y=peak, color='r', linestyle='--', alpha=0.5)

# Annotate peaks as diastolic and systolic alternately
peak_type = ['Diastolic Peak', 'Systolic Peak']
peak_color = ['green', 'red']
for i in range(len(peaks)):
    plt.annotate(peak_type[i % 2], (peaks[i], ppg_signal_subset[peaks[i]]),
                 textcoords="offset points", xytext=(0, 10), ha='center', color=peak_color[i % 2])

# Add labels and legend
plt.title('PPG Waveform')
plt.xlabel('Index')
plt.ylabel('Signal Value')
plt.yticks([])  # Remove y-axis values
plt.legend()
plt.grid(True)

# Get the script directory (assuming the script name is main.py)
import os
script_dir = os.path.dirname(__file__)

# Save the plot with a default name (finalplot.png)
save_path = os.path.join(script_dir, 'finalplot.png')
plt.savefig(save_path)

# Show the plot (optional)
# plt.show()
