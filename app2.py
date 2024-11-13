# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 2em; 
            color: #4A90E2; 
            font-weight: bold; 
            text-align: center;
        }
        .section-header {
            font-size: 1.3em; 
            font-weight: bold; 
            color: #4A4A4A;
            margin-top: 20px;
        }
        .input-field {
            padding: 10px;
        }
        .predict-button {
            background-color: #4A90E2;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .predict-button:hover {
            background-color: #357ABD;
        }
        .result-positive {
            color: #D9534F;
            font-weight: bold;
            font-size: 1.2em;
            text-align: center;
        }
        .result-negative {
            color: #5CB85C;
            font-weight: bold;
            font-size: 1.2em;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Student Mental health.csv')
    data.drop(columns=['Timestamp'], inplace=True)  # Drop irrelevant column
    data['Age'].fillna(data['Age'].median(), inplace=True)  # Fill missing Age values with median

    # Encode categorical variables
    encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = encoder.fit_transform(data[column])
    return data

# Train the model and return the best model
@st.cache_resource
def train_model(data):
    X = data.drop(['Do you have Depression?'], axis=1)  # Features
    y = data['Do you have Depression?']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC()
    }

    best_model = None
    best_accuracy = 0
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model

# Load data and train model
data = load_data()
model = train_model(data)

# Streamlit App Layout
st.markdown('<div class="title">Mental Health Prediction</div>', unsafe_allow_html=True)
st.write("Fill in the information below to check for potential signs of depression:")

st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ("Female", "Male"))
    age = st.number_input("Age", min_value=10, max_value=100, value=20)

with col2:
    course = st.selectbox("Course", ("Engineering", "Others"))
    year_of_study = st.slider("Year of Study", 1, 5, 2)

st.markdown('<div class="section-header">Academic and Personal Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
    marital_status = st.selectbox("Marital Status", ("No", "Yes"))

with col4:
    anxiety = st.selectbox("Do you have Anxiety?", ("No", "Yes"))
    panic_attack = st.selectbox("Do you have Panic Attack?", ("No", "Yes"))
    treatment = st.selectbox("Have you sought treatment?", ("No", "Yes"))

# Convert inputs to numeric values
gender = 0 if gender == "Female" else 1
course = 0 if course == "Engineering" else 1
marital_status = 0 if marital_status == "No" else 1
anxiety = 0 if anxiety == "No" else 1
panic_attack = 0 if panic_attack == "No" else 1
treatment = 0 if treatment == "No" else 1

# Prepare input array for prediction
input_data = np.array([[gender, age, course, year_of_study, cgpa, marital_status, anxiety, panic_attack, treatment]])

# Prediction Button and Result
if st.button("Predict", key="predict", help="Click to predict", use_container_width=True):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.markdown('<div class="result-positive">⚠️ The model predicts that you may have depression.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-negative">✅ The model predicts that you do not have depression.</div>', unsafe_allow_html=True)
