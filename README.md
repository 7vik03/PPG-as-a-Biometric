# PPG-as-a-Biometric
# PPG as a Biometric: Evaluating Statistical ML Models in Low-Fidelity Scenarios

This repository implements and evaluates a machine learning pipeline for biometric authentication using **photoplethysmography (PPG) signals**. The project supports our research on leveraging simple statistical features to classify individuals based on their PPG patterns — even under constrained preprocessing and signal quality.

> **"PPG as a Biometric: A Study on the Effectiveness of Statistical Input-Based ML Algorithms in Disadvantageous Scenarios"**

---

## 🧠 Project Motivation

PPG signals, typically obtained from devices like smartwatches, are a promising biometric due to their liveness detection and uniqueness. However, many existing systems depend on high-quality signals and complex feature engineering. This project explores whether **lightweight statistical descriptors** (mean, median, skew, kurtosis) can still deliver high classification performance — enabling applications in **edge computing** and **real-world noise scenarios**.

---

## 📁 Project Structure
.
├── data/
│ ├── test8.csv # Raw dataset
│ ├── merged_data.csv # All patient signals merged
│ └── stat_modified_data.csv # Final dataset with extracted statistical features
│
├── src/
│ ├── preprocessing/
│ │ ├── new_column.py
│ │ ├── newfinalmerge.py
│ │ └── stats.py
│ ├── training/
│ │ ├── gbmfile.py
│ │ ├── randomforest_test.py
│ │ ├── kmeansclus.py
│ │ └── rocfile.py
│ └── visualization/
│ ├── waveform.py
│ ├── finalplot.png
│ ├── comparison_bar_graph.png
│ └── comparison_bar_graph_with_time.png


---

## 📊 Dataset Overview

- **Original File**: `test8.csv`
- **Participants**: 35 individuals
- **Samples per signal**: 300
- **Sampling rate**: 50 Hz
- **Final shape**: ~87,500 samples with `mean`, `median`, `skew`, `kurtosis` derived for each 25-point window

---

## 🧮 Feature Extraction Pipeline

Features extracted using a fixed window (25 samples):
- `signal_mean_diff`
- `signal_median_diff`
- `signal_kurtosis_diff`
- `signal_skewness`

→ Final output saved as: `stat_modified_data.csv`

---

## 🤖 Model Training & Results

| Model               | Accuracy | F1 Score | Cross-Validation |
|--------------------|----------|----------|------------------|
| **Gradient Boosting (XGBoost)** | 90.27%   | 90.22%   | 89.15%           |
| Random Forest       | 83.00%   | 82.50%   | 81.06%           |
| K-Nearest Neighbors | 67.00%   | 66.40%   | 65.76%           |

Evaluation used:
- 5-fold cross-validation
- Weighted F1 Score
- ROC AUC (micro & macro)

---

## 📈 Visualizations

### 🔹 Model Performance
![Accuracy & F1](src/visualization/comparison_bar_graph.png)

### 🔹 Training Time Comparison
![Training Time](src/visualization/comparison_bar_graph_with_time.png)

### 🔹 PPG Waveform with Annotated Peaks
![Waveform](src/visualization/finalplot.png)

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python src/preprocessing/new_column.py
python src/preprocessing/newfinalmerge.py
python src/preprocessing/stats.py
python src/training/gbmfile.py
python src/training/randomforest_test.py
python src/training/kmeansclus.py
python src/training/rocfile.py
