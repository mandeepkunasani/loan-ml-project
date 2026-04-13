# ==============================
# LOAN PREDICTION ML PROJECT
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("STARTING LOAN PROJECT...")

# 🔥 IMPORTANT: use FULL PATH (change if needed)
df = pd.read_csv("/Users/kunasanimandeep/Downloads/loan_data_set.csv")

print("Shape:", df.shape)

# Drop unnecessary column
df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical to numeric
df.replace({
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
    'Loan_Status': {'Y': 1, 'N': 0}
}, inplace=True)

# Fix Dependents
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Ensure target is numeric
df['Loan_Status'] = df['Loan_Status'].astype(int)

# Split data
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("✅ DONE")