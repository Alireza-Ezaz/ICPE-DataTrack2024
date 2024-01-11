# This script trains a Random Forest classifier to detect anomalies in specific microservices.
# It then prints a classification report to evaluate the model's performance.
# The classification report includes precision, recall, and F1-score for each class.

"""
Precision: The ratio of true positive predictions to the total number of positive predictions for each class.
A high precision means that when the model predicts an anomaly for a microservice, it is likely correct.

Recall: The ratio of true positive predictions to the total number of actual positives for each class.
A high recall means that the model is good at catching all the anomalies for a microservice.

F1-score: The harmonic mean of precision and recall, providing a single score that balances both concerns.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Set the directory path where your CSV files are located
directory_path = '../../tracing-data.tar/tracing-data/social-network/'

# Get a list of CSV files in the specified directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Load each CSV file, label it with the appropriate microservice, and concatenate them
for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    data = pd.read_csv(file_path).head(10000)
    # The label is the file name without the extension or 'no-interference' for the no anomaly case
    label = csv_file.replace('.csv', '') if 'no-interference' not in csv_file else 'no_anomaly'
    data['anomaly_injected'] = label
    all_data = pd.concat([all_data, data], ignore_index=True)

# Prepare features and labels
feature_cols = all_data.columns[1:-1]  # Exclude 'trace_id' and 'anomaly_injected'
X = all_data[feature_cols]
y = all_data['anomaly_injected']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(15, 5))
plt.title('Feature Importances')
bars = plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [feature_cols[i][:2] for i in indices], rotation=90)
plt.xlim([-1, len(indices)])

# Add labels to the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, round(yval, 4), va='bottom')  # Adjust the position of the text if needed

plt.show()
