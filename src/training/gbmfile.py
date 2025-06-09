import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import numpy as np
import pickle

# Provide the full path to the file
data = pd.read_csv('stat_modified_data.csv')

# Split the data into features and target
X = data.drop(columns=['target_patient'])
y = data['target_patient'] - 1

# Check the shapes of X and y arrays
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Reshape y to have a shape of (87500, 1)
y = y.values.reshape(-1, 1)

# Concatenate X and y arrays along the row axis
data_array = np.concatenate((X.values, y), axis=1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_array[:, :-1], data_array[:, -1], test_size=0.34, random_state=42)

# Define the GBM model
gbm_model = xgb.XGBClassifier(objective='multi:softprob', num_class=35, max_depth=6, learning_rate=0.1, n_estimators=1000, n_jobs=-1)

# Train the model
gbm_model.fit(X_train, y_train)

# Print the accuracy of the model
y_pred = gbm_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Print the F1 score of the model
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 score:", f1)

# Print the cross-validation score of the model
cv_scores = cross_val_score(gbm_model, X_train, y_train, cv=5, scoring='f1_weighted')
print("Cross-validation score:", np.mean(cv_scores))

# Save the trained model
with open('gbmmodel.pkl', 'wb') as f:
    pickle.dump(gbm_model, f)