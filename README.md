# 💰 Loan Approval Predictor 

A **Machine Learning-based web app** that predicts loan approval status with **88% accuracy**, using applicant details like income, credit history, and personal information. Built with Python and Streamlit for interactive real-time predictions.  

[🌐 Live Demo](https://meenakshi-1802.streamlit.app/)

---

## 🧠 Machine Learning Highlights

- **Model:** Random Forest Classifier  
- **Accuracy:** **88%** on test data  
- **Techniques Used:**  
  - Handling missing values & outliers  
  - Feature engineering (Total Income, Loan-Income Ratio, Log transforms)  
  - Encoding categorical variables (LabelEncoder)  
  - Feature scaling (StandardScaler)  
  - Class imbalance handling (SMOTE)  
- **Saved Model & Preprocessing Pipelines:**  
  - `loan_model.pkl`  
  - `scaler.pkl`  
  - `encoders.pkl`  

---

## 🔹 Features

- Predicts **Loan Approved ✅** or **Loan Denied ❌**  
- Shows **approval probability** for transparency  
- Clean, interactive **Streamlit interface**  
- Light/Dark theme toggle 🔆  
- GitHub profile footer integration  

---

## 📊 Dataset

- Source: Public loan prediction datasets  
- Key columns:  
  - `ApplicantIncome`, `CoapplicantIncome`  
  - `LoanAmount`, `Loan_Amount_Term`  
  - `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`  
  - `Credit_History`, `Property_Area`  
  - Target: `Loan_Status`  

---

## 🛠 Technology & Tools

- **Python:** Pandas, NumPy, Matplotlib, Seaborn  
- **Scikit-learn:** Logistic Regression, Random Forest, Gradient Boosting  
- **Imbalanced-learn:** SMOTE  
- **Joblib:** Model & encoder serialization  
- **Streamlit:** Web application  

---

## 🚀 How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/Meenakshi-1802/Loan_Approval_Predictor.git
cd Loan_Approval_Predictor

2. **Install dependencies**

```bash
pip install -r requirements.txt

3. **Run the app**

```bash
streamlit run app/app.py
4. **Open the URL shown in the terminal**


