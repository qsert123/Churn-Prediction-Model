import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report

#load the dataset
data = pd.read_csv('E:\Projects\streamlit\project1\data\cleaned_data.csv') 
df = pd.DataFrame(data)

#data preprocessing 
y = df['Churn'] 
x = df.drop(['Churn'], axis=1)

#feature scaling 
scaler = StandardScaler() 
x_scaled = scaler.fit_transform(x)

#split the train and test data 
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

#train the model 
model = LogisticRegression() 
model.fit(x_train, y_train) 

#predict 
y_pred = model.predict(x_test)

#evaluation 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy:", accuracy) 
print(classification_report(y_test, y_pred))
