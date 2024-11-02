import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Liver Cancer Prediction App", layout="wide")

# Load the CSV data
data = pd.read_csv("data.csv")

# Extract the numeric columns for regression
numeric_columns = ["Age", "Grams_day", "Packs_year","INR", "AFP", "Hemoglobin", "MCV", "Leucocytes", "Platelets", "Albumin", "Total_Bil", "ALT", "AST", "GGT", "ALP", "TP", "Creatinine", "Nodule", "Major_Dim", "Dir_Bil", "Iron", "Sat", "Ferritin", "survivalTarget" ]
numeric_data = data[numeric_columns]

# Extract the categorical columns for plotting
categorical_columns = ["Gender", "Symtoms", "Alcohol", "HBsAg", "HBeAg", "HBcAg", "HCVAb", "Cirrhosis", "Endemic", "Smoking", "Diabetes", "Obesity", "Hemochro", "AHT", "CRI", "HIV", "NASH", "Varices", "Spleno", "PHT", "PVT", "Metastasis", "Hallmark"]

# Compute the median age
median_age = numeric_data["Age"].median()

# Replace '?' with median value in each column
for column in numeric_data.columns:
    numeric_data[column] = pd.to_numeric(numeric_data[column], errors='coerce')
    median_value = numeric_data[column].median()
    numeric_data[column] = numeric_data[column].fillna(median_value)

# Train the initial regression model
X = numeric_data.drop("survivalTarget", axis=1)
y = numeric_data["survivalTarget"].astype(int) # Convert survivalTarget to integer
regression_model = LinearRegression()
regression_model.fit(X, y)

# Plot the initial regression line in black
plt.scatter(X["Age"], y, color="black", label="Data Point")
plt.scatter(X["Age"], regression_model.predict(X), color="grey", label="Default", marker='o')

# Streamlit App
st.title("การคาดการณ์โอกาสรอดของมะเร็งตับ")
st.write("ฝึกจากชุดข้อมูลจำนวน 165 รายผู้ป่วยทั้งที่รอดและเสียชีวิต")

# User Inputs
age = st.slider("อายุ", int(numeric_data["Age"].min()), 100, int(median_age))

# Modify the features based on user input
modified_X = X.copy()
modified_X.loc[0, "Age"] = age

modifiable_features = ["Packs_year", "Hemoglobin", "Albumin", "Creatinine", "INR", "Platelets", "Nodule"]

for feature in modifiable_features:
    if feature == "Packs_year":
        value = st.slider(feature, 0, 100)
    elif feature =="Hemoglobin" : 
        value = st.slider(feature,3,18)
    elif feature =="Albumin" : 
        value = st.slider(feature,0,5)
    elif feature =="Creatinine" : 
        value = st.slider(feature,0,10)
    elif feature =="INR" : 
        value = st.slider(feature,1,10)
    elif feature =="Platelets" : 
        value = st.slider(feature, 10, 500000)
    else:
        value = st.slider(feature, 0, 5)
    modified_X.loc[0, feature] = value

# Predictions
initial_prediction = regression_model.predict(X)
modified_prediction = regression_model.predict(modified_X)

# Plot the second regression line in blue
plt.scatter(X["Age"], initial_prediction, color="blue", label="Median Case", marker='+')

# Plot the third regression line in red or green
if modified_prediction[0] < initial_prediction[0]:
    plt.scatter(modified_X["Age"], modified_prediction, color="red", label="Worse", marker='*')
else:
    plt.scatter(modified_X["Age"], modified_prediction, color="green", label="Better", marker='*')

# Plot settings
plt.xlabel("Age")
plt.ylabel("Survival Probability")
plt.legend()
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1

# Show the plot
st.pyplot(plt.gcf())

# Accuracy metrics
y_pred = regression_model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write(f"Mean Squared Error: {mse}")
st.write(f"R^2 Score: {r2}")

