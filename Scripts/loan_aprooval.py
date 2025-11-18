import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap

# Load dataset
df = pd.read_csv("data/loan_prediction.csv")
print("Data Loaded Successfully!")
print(df.head())
print(df.isnull().sum())

# Clean data (avoid chained assignment warning)
df = df.copy()
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Feature engineering
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Loan_Income_Ratio'] = df['LoanAmount'] / df['Total_Income']
df['Has_Dependents'] = df['Dependents'].apply(lambda x: 1 if x > 0 else 0)
df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)
df['Total_Income_log'] = np.log(df['Total_Income'] + 1)

# Encode Categorical Variables
categorical_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "Scripts/encoders.pkl")

# Visualizations 
numeric_df = df.select_dtypes(include=np.number)  # select only numeric columns

sns.countplot(x='Loan_Status', data=df)
plt.show()

sns.histplot(df['Total_Income_log'], bins=30, kde=True)
plt.show()

sns.boxplot(x='Education', y='LoanAmount_log', data=df)
plt.show()

sns.countplot(x='Property_Area', hue='Loan_Status', data=df)
plt.show()

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()


# Prepare Features & Target
X = df.drop(['Loan_ID','Loan_Status'], axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "Scripts/scaler.pkl")

# Train & Compare Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_res)
    y_pred = model.predict(X_test_scaled)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Save Best Model (Random Forest)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train_scaled, y_train_res)
joblib.dump(best_model, "Scripts/loan_model.pkl")
print("Model saved successfully!")

# Predict New Applicant
# Use original categorical strings here
new_applicant = pd.DataFrame([[
    "Male",       # Gender
    "Yes",        # Married
    1,            # Dependents
    "Graduate",   # Education
    "No",         # Self_Employed
    5000,         # ApplicantIncome
    2000,         # CoapplicantIncome
    150,          # LoanAmount
    360,          # Loan_Amount_Term
    1,            # Credit_History
    "Urban"       # Property_Area
]],
columns=['Gender','Married','Dependents','Education','Self_Employed',
         'ApplicantIncome','CoapplicantIncome','LoanAmount',
         'Loan_Amount_Term','Credit_History','Property_Area'])

# Feature engineering for new applicant
new_applicant['Total_Income'] = new_applicant['ApplicantIncome'] + new_applicant['CoapplicantIncome']
new_applicant['Loan_Income_Ratio'] = new_applicant['LoanAmount'] / new_applicant['Total_Income']
new_applicant['Has_Dependents'] = new_applicant['Dependents'].apply(lambda x: 1 if x>0 else 0)
new_applicant['LoanAmount_log'] = np.log(new_applicant['LoanAmount'] + 1)
new_applicant['Total_Income_log'] = np.log(new_applicant['Total_Income'] + 1)

# Load encoders & scaler
model = joblib.load("Scripts/loan_model.pkl")
scaler = joblib.load("Scripts/scaler.pkl")
encoders = joblib.load("Scripts/encoders.pkl")

# Encode categorical columns for new applicant
for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
    le = encoders[col]
    new_applicant[col] = le.transform(new_applicant[col])

# Scale and predict
new_applicant_scaled = scaler.transform(new_applicant)
prediction = model.predict(new_applicant_scaled)
prediction_proba = model.predict_proba(new_applicant_scaled)[0][1]  # Probability of approval

print("Loan Approved ✅" if prediction[0]==1 else "Loan Not Approved ❌")
print(f"Approval Probability: {prediction_proba*100:.2f}%")
