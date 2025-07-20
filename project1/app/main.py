
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("../data/cleaned_data.csv")

# Data preprocessing
y = df['Churn']
x = df.drop(['Churn'], axis=1)

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the train and test data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)

st.title("Churn Prediction Model")

# Sidebar for custom input
st.sidebar.header("Predict Churn for Custom Input")
user_input = {}
for col in x.columns:
    if col.lower() == "gender":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=Female, 1=Male)", options=[0, 1])
    elif col.lower() == "seniorcitizen":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=No, 1=Yes)", options=[0, 1])
    elif col.lower() == "partner":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=No, 1=Yes)", options=[0, 1])
    elif col.lower() == "dependents":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=No, 1=Yes)", options=[0, 1])
    elif col.lower() == "phoneservice":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=No, 1=Yes)", options=[0, 1])
    elif col.lower() == "multiplelines":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=No Phone Service, 1=No, 2=Yes)", options=[0, 1, 2])
    elif col.lower() == "internetservice":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=DSL, 1=Fibre, 2=No)", options=[0, 1, 2])
    elif col.lower() == "contract":
        user_input[col] = st.sidebar.selectbox(f"{col} (0=per/month, 1=one/yr, 2=Two/yr)", options=[0, 1, 2])
    elif pd.api.types.is_numeric_dtype(x[col]):
        user_input[col] = st.sidebar.number_input(f"{col}", float(x[col].min()), float(x[col].max()), float(x[col].mean()))
    else:
        user_input[col] = st.sidebar.selectbox(f"{col}", options=x[col].unique())
 
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
user_pred = model.predict(input_scaled)[0]
user_pred_proba = model.predict_proba(input_scaled)[0][1]

st.sidebar.subheader("Prediction for Custom Input")
st.sidebar.write(f"Churn Prediction: {'Yes' if user_pred else 'No'}")
st.sidebar.write(f"Churn Probability: {user_pred_proba:.2f}")

# Central graph updates dynamically
st.subheader("Custom Input Feature Values")
fig3, ax3 = plt.subplots()
input_df.T.plot(kind='bar', legend=False, ax=ax3)
ax3.set_ylabel("Value")
ax3.set_title("Custom Input Features")
st.pyplot(fig3)

st.subheader("Churn Probability for Custom Input")
st.progress(user_pred_proba)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")
st.subheader("Classification Report")
st.text(report_text)
