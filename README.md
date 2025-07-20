# Churn-Prediction-Model
The Project implements a Churn prediction using Logistic Regression, which analyze customer patterns to predict the risk of leaving Customer
## 📂 Project Structure
├── app 
  └── main.py 
├── data
  └──  cleaned_data.csv 
  └──  data.ipynb              
  └──  WA_Fn-UseC_-Telco...    # Raw dataset
├── model
  └── test.py                 # Churn prediction model script

---

## 🚀 Features

- 📊 Churn prediction using Logistic Regression
- ⚖️ Bias balancing with `class_weight='balanced'`
- 🧮 Model evaluation with accuracy, precision, recall, F1 score
- 📈 Visualizations: Confusion matrix, prediction distribution
- 🌐 Streamlit dashboard for user-friendly interface

---

## 🧠 Model Overview

- **Algorithm**: Logistic Regression  
- **Data Preprocessing**: StandardScaler  
- **Train-Test Split**: 80/20 with stratification  
- **Evaluation Metrics**:
  - Accuracy
  - Classification Report
  - Confusion Matrix

---
## Run the app
  - streamlit run app.py
---


## ⚙️ Installation

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt


