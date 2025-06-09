import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import pickle

# Load the dataset
data = pd.read_csv('stat_modified_data.csv')

# Split the data into features and target
X = data.drop(columns=['target_patient'])
y = data['target_patient']

# Binarize the output labels for ROC curve
y_bin = label_binarize(y, classes=range(1, len(y.unique()) + 1))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)
y_test_bin = label_binarize(y_test, classes=range(1, len(y.unique()) + 1))

# 1. KNN
print("K-Nearest Neighbors (KNN)")

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Perform cross-validation
cv_scores_knn = cross_val_score(knn_classifier, X_train, y_train, cv=5)
print("Cross-validation Scores (KNN):", cv_scores_knn)
print("Mean CV Accuracy (KNN):", cv_scores_knn.mean())

# Fit the model
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_classifier.predict(X_test)

# Print classification report
print("Classification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

# Calculate and print F1 score
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
print("F1 Score (KNN):", f1_knn)

# Predict probabilities for ROC curve
y_prob_knn = knn_classifier.predict_proba(X_test)

# Calculate ROC curve and ROC area for each class
fpr_knn = dict()
tpr_knn = dict()
roc_auc_knn = dict()

for i in range(y_test_bin.shape[1]):
    fpr_knn[i], tpr_knn[i], _ = roc_curve(y_test_bin[:, i], y_prob_knn[:, i])
    roc_auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])

# Compute micro-average ROC curve and ROC area
fpr_knn["micro"], tpr_knn["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob_knn.ravel())
roc_auc_knn["micro"] = auc(fpr_knn["micro"], tpr_knn["micro"])

# Compute macro-average ROC curve and ROC area
all_fpr_knn = np.unique(np.concatenate([fpr_knn[i] for i in range(y_test_bin.shape[1])]))
mean_tpr_knn = np.zeros_like(all_fpr_knn)
for i in range(y_test_bin.shape[1]):
    mean_tpr_knn += np.interp(all_fpr_knn, fpr_knn[i], tpr_knn[i])
mean_tpr_knn /= y_test_bin.shape[1]
fpr_knn["macro"] = all_fpr_knn
tpr_knn["macro"] = mean_tpr_knn
roc_auc_knn["macro"] = auc(fpr_knn["macro"], tpr_knn["macro"])

# Plot the micro-average ROC curve
plt.figure()
plt.plot(fpr_knn["micro"], tpr_knn["micro"], color='deeppink', linestyle=':', linewidth=4,
         label=f'micro-average ROC curve (area = {roc_auc_knn["micro"]:0.2f})')

# Plot the macro-average ROC curve
plt.plot(fpr_knn["macro"], tpr_knn["macro"], color='navy', linestyle=':', linewidth=4,
         label=f'macro-average ROC curve (area = {roc_auc_knn["macro"]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (KNN)')
plt.legend(loc='lower right')
plt.show()

# 2. Random Forest
print("Random Forest")

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Perform cross-validation
cv_scores_rf = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print("Cross-validation Scores (RF):", cv_scores_rf)
print("Mean CV Accuracy (RF):", cv_scores_rf.mean())

# Fit the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test)

# Print classification report
print("Classification Report (RF):")
print(classification_report(y_test, y_pred_rf))

# Calculate and print F1 score
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print("F1 Score (RF):", f1_rf)

# 3. GBM
print("Gradient Boosting Machine (GBM)")

# Adjust the target for GBM
y_gbm = y - 1

# Split the data for GBM
X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(X.values, y_gbm.values, test_size=0.34, random_state=42)

# Define the GBM model
gbm_model = xgb.XGBClassifier(objective='multi:softprob', num_class=35, max_depth=6, learning_rate=0.1, n_estimators=1000, n_jobs=-1)

# Train the model
gbm_model.fit(X_train_gbm, y_train_gbm)

# Make predictions
y_pred_gbm = gbm_model.predict(X_test_gbm)

# Print the accuracy of the model
accuracy_gbm = np.mean(y_pred_gbm == y_test_gbm)
print("Accuracy (GBM):", accuracy_gbm)

# Print the F1 score of the model
f1_gbm = f1_score(y_test_gbm, y_pred_gbm, average='weighted')
print("F1 Score (GBM):", f1_gbm)

# Print the cross-validation score of the model
cv_scores_gbm = cross_val_score(gbm_model, X_train_gbm, y_train_gbm, cv=5, scoring='f1_weighted')
print("Cross-validation score (GBM):", np.mean(cv_scores_gbm))

# Save the trained model
with open('gbmmodel.pkl', 'wb') as f:
    pickle.dump(gbm_model, f)