import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the datasets
train_transaction = pd.read_csv('/Users/aaryanpatel/Desktop/train_transaction.csv')

# Assuming 'isFraud' is the target variable
X_train = train_transaction.drop('isFraud', axis=1)
y_train = train_transaction['isFraud']

# Filling missing values for numeric columns with their mean
X_train_numeric = X_train.select_dtypes(include=['int64', 'float64'])
X_train_numeric = X_train_numeric.fillna(X_train_numeric.mean())

# Encoding categorical variables
X_train_categorical = X_train.select_dtypes(include=['object']).apply(LabelEncoder().fit_transform)

# Combining numeric and encoded categorical variables
X_train_processed = pd.concat([X_train_numeric, X_train_categorical], axis=1)

# Spliting the data into two different train/test splits
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X_train_processed, y_train, test_size=0.5, random_state=42)
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

# Defining the XGBoost model
model_50 = xgb.XGBClassifier()
model_80 = xgb.XGBClassifier()


model_50.fit(X_train_50, y_train_50)
model_80.fit(X_train_80, y_train_80)


y_pred_50 = model_50.predict(X_test_50)
y_pred_80 = model_80.predict(X_test_80)


accuracy_50 = accuracy_score(y_test_50, y_pred_50)
roc_auc_50 = roc_auc_score(y_test_50, y_pred_50)
accuracy_80 = accuracy_score(y_test_80, y_pred_80)
roc_auc_80 = roc_auc_score(y_test_80, y_pred_80)


print(f"50/50 Split - Accuracy: {accuracy_50:.2f}, ROC AUC: {roc_auc_50:.2f}")
print(f"80/20 Split - Accuracy: {accuracy_80:.2f}, ROC AUC: {roc_auc_80:.2f}")


#2



'''Selecting a subset of columns for the worse model
For example, you could use just 'TransactionID', 'TransactionDT', and a couple of 'V' columns'''
subset_columns = ['TransactionID', 'TransactionDT', 'V1', 'V2']

# Creating the dataset using just the subset of columns
X_train_subset = X_train_processed[subset_columns]

# Spliting the data for the worse model
X_train_subset_50, X_test_subset_50, y_train_subset_50, y_test_subset_50 = train_test_split(
    X_train_subset, y_train, test_size=0.5, random_state=42)

# Defining the worse XGBoost model
model_subset_50 = xgb.XGBClassifier()

# Training the worse model on the 50/50 split
model_subset_50.fit(X_train_subset_50, y_train_subset_50)

# Predicting on the test set
y_pred_subset_50 = model_subset_50.predict(X_test_subset_50)

# Calculating accuracy and ROC AUC for the worse model
accuracy_subset_50 = accuracy_score(y_test_subset_50, y_pred_subset_50)
roc_auc_subset_50 = roc_auc_score(y_test_subset_50, y_pred_subset_50)


print(f"Worse Model 50/50 Split - Accuracy: {accuracy_subset_50:.2f}, ROC AUC: {roc_auc_subset_50:.2f}")







''' Assuming your model's predict_proba method returns a 2D array, where the second column 
represents the probability of the positive class. If not, adjust the indexing accordingly.'''
y_pred_probs_50 = model_50.predict_proba(X_test_50)[:, 1]
y_pred_probs_subset_50 = model_subset_50.predict_proba(X_test_subset_50)[:, 1]

# Calculating true positive rate and the corresponding thresholds for both models
fpr_50, tpr_50, thresholds_50 = roc_curve(y_test_50, y_pred_probs_50)
fpr_subset_50, tpr_subset_50, thresholds_subset_50 = roc_curve(y_test_subset_50, y_pred_probs_subset_50)

# Calculating the cumulative gains
cumulative_gain_50 = tpr_50 / (fpr_50 + tpr_50)
cumulative_gain_subset_50 = tpr_subset_50 / (fpr_subset_50 + tpr_subset_50)

# Ploting the gains curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_50, cumulative_gain_50, label='Model 1')
plt.plot(fpr_subset_50, cumulative_gain_subset_50, label='Model 2', linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', label='Random Targeting')

plt.title('Incremental Gains Chart')
plt.xlabel('Proportion of population targeted')
plt.ylabel('Cumulative Extra Sales')
plt.legend(loc='best')

plt.show()
