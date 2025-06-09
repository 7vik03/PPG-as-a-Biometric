# PPG-as-a-Biometric
# PPG as a Biometric: Evaluating Statistical ML Models in Low-Fidelity Scenarios

This repository contains a complete machine learning pipeline for biometric authentication using photoplethysmography (PPG) signals. It demonstrates that simple statistical features can perform well even under constrained signal quality and minimal preprocessing, aligning with the findings of our research:

> **"PPG as a Biometric: A Study on the Effectiveness of Statistical Input-Based ML Algorithms in Disadvantageous Scenarios"**

---

## 🧠 Motivation

PPG signals are increasingly used in biometric systems due to their ability to provide liveness detection and user uniqueness. While existing methods rely on signal denoising, transformation, and complex processing, this project evaluates the effectiveness of **mean**, **median**, **skew**, and **kurtosis** features under intentionally suboptimal preprocessing — useful for **edge deployments**.

---

## 📁 Project Structure

```
.
├── data/
│   ├── test8.csv
│   ├── merged_data.csv
│   └── stat_modified_data.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── new_column.py
│   │   ├── newfinalmerge.py
│   │   └── stats.py
│   ├── training/
│   │   ├── gbmfile.py
│   │   ├── randomforest_test.py
│   │   ├── kmeansclus.py
│   │   └── rocfile.py
│   └── visualization/
│       ├── waveform.py
│       ├── finalplot.png
│       ├── comparison_bar_graph.png
│       └── comparison_bar_graph_with_time.png
```

---

## 📊 Dataset

- **Source**: test8.csv
- **Patients**: 35
- **Signals per patient**: ~50–60
- **Samples per signal**: 300
- **Sampling Rate**: 50 Hz

---

## 🧮 Feature Extraction

From every 25-point signal window:
- `signal_mean_diff`
- `signal_median_diff`
- `signal_kurtosis_diff`
- `signal_skewness`

Saved to: `stat_modified_data.csv`

---

## 🤖 Model Performance

| Model               | Accuracy | F1 Score | Cross-Validation |
|--------------------|----------|----------|------------------|
| XGBoost (GBM)       | 90.27%   | 90.22%   | 89.15%           |
| Random Forest       | 83.00%   | 82.50%   | 81.06%           |
| K-Nearest Neighbors | 67.00%   | 66.40%   | 65.76%           |

---

## 📊 Visual Outputs

You can view the following plots in `src/visualization/`:
- Model performance comparison: `comparison_bar_graph.png`
- Training time vs accuracy tradeoff: `comparison_bar_graph_with_time.png`
- Annotated waveform plot: `finalplot.png`

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python src/preprocessing/new_column.py
python src/preprocessing/newfinalmerge.py
python src/preprocessing/stats.py
```

### 3. Train Models
```bash
python src/training/gbmfile.py
python src/training/randomforest_test.py
python src/training/kmeansclus.py
python src/training/rocfile.py
```

---

## 📝 Future Improvements

- Adaptive windowing strategies
- Systolic/diastolic peak detection features
- Real-time biometric deployment using r-PPG
- On-device lightweight model compression for wearables

---

## 👨‍🔬 Author

Created by **Venkata Sai Sathvik Rajampalli**  
Licensed under the [MIT License](LICENSE)

---

## 📚 Citation

```bibtex
@misc{rajampalli2024ppgbiometric,
  title={PPG as a Biometric: A Study on the Effectiveness of Statistical Input-Based ML Algorithms in Disadvantageous Scenarios},
  author={Rajampalli, Venkata Sai Sathvik and Dharshini, Kaviya and Jayaseelan, Nithillen and Jeeva, J.B.},
  year={2024},
  note={Conference: iCASIC, VIT Vellore}
}
```
