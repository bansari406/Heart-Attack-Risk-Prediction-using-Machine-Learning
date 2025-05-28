import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


Data_Set = pd.read_csv(r"C:/Bushra/Python Notes/Data_set/Medicaldataset.csv")
print(Data_Set.head())
print(Data_Set.info())
print(Data_Set.describe())


#Convert the result into binary values - Gender is Target Variable
# Strip spaces and convert to lowercase
Data_Set["Result"] = Data_Set["Result"].str.strip().str.lower()

# # Then apply the mapping
Data_Set['Result'] = Data_Set['Result'].map({'negative': 0, 'positive': 1})
print(Data_Set["Gender"].value_counts())

# Boold Sugar
Data_Set["Blood sugar"] = Data_Set["Blood sugar"].apply(lambda x: 1 if x > 120 else 0)

# --- Handle Outliers ---
# Cap heart rate at 200 bpm
Data_Set['Heart rate'] = np.where(Data_Set['Heart rate'] > 200, 200, Data_Set['Heart rate'])

# Cap CK-MB at 100 (domain knowledge or 99th percentile could be used)
Data_Set['CK-MB'] = np.where(Data_Set['CK-MB'] > 100, 100, Data_Set['CK-MB'])

# Cap Blood Sugar at 300
Data_Set['Blood sugar'] = np.where(Data_Set['Blood sugar'] > 300, 300, Data_Set['Blood sugar'])

# --- Log transformation of skewed features ---
# Add small constant to avoid log(0)
Data_Set['CK-MB_log'] = np.log(Data_Set['CK-MB'] + 1)
Data_Set['Troponin_log'] = np.log(Data_Set['Troponin'] + 1)

# --- Correlation Heatmap ---
# Select relevant columns for correlation
corr_cols = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 
             'Diastolic blood pressure', 'Blood sugar', 'CK-MB_log', 'Troponin_log', 'Result']

plt.figure(figsize=(10, 8))
sns.heatmap(Data_Set[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Step 4: Feature Selection
features = ['Age', 'Gender', 'Systolic blood pressure', 'Diastolic blood pressure','Blood sugar', 'CK-MB_log', 'Troponin_log']
x = Data_Set[features]
y = Data_Set['Result']

# X = Data_Set.drop("Result", axis=1)
# y = Data_Set["Result"]

#Train Split 80/20
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# print("x_train shape:", x_train.shape)
# print("x_test shape:", x_test.shape)

# Random Forest Classifier (Train the model) 80% of the data
model = RandomForestClassifier( n_estimators=100, random_state=42)
model.fit(x_train,y_train)

# Make prediction by evaluating the model, 20% of the data
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))

# Feature Analysis (Why Troponin, CK-MB, BP Matter)
importance = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance)

plt.figure(figsize=(10,6))
plt.barh(importance['Feature'], importance['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.xlabel("Random Forest Importance Score")
plt.show()

# Confusion Matrix

cm_matrix = confusion_matrix(y_test,y_pred)
print("Confusion matrix:", cm_matrix)
plt.figure(figsize=(8,6))
sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Logistic Regression Model
Log_model = LogisticRegression(max_iter=1000)
Log_model.fit(x_train,y_train)
plt.figure(figsize=(10,6))
y_pred_log = Log_model.predict(x_test)
print("Logistics Regression Accuracy:", accuracy_score(y_test,y_pred_log))
# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

# Confusion matrix
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Greens')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



