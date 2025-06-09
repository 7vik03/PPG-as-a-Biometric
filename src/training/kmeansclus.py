import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

# Load the dataset
data = pd.read_csv('stat_modified_data.csv')

# Split the data into features and target
X = data.drop(columns=['target_patient'])
y = data['target_patient']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Perform cross-validation
cv_scores = cross_val_score(knn_classifier, X_train, y_train, cv=5)

# Print cross-validation scores
print("Cross-validation Scores (KNN):", cv_scores)
print("Mean CV Accuracy (KNN):", cv_scores.mean())

# Fit the model
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Print classification report
print("Classification Report (KNN):")
print(classification_report(y_test, y_pred))

# Calculate and print F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score (KNN):", f1)
