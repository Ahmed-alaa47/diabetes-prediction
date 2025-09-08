# diabetes-prediction

This project is a Machine Learning web application built with Streamlit that predicts whether a person has diabetes based on medical features.
Users can enter their health data, choose between two trained models (Logistic Regression and Random Forest), and instantly see the prediction result.

# Dataset & Features

The app uses six key features for prediction:
Glucose
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age

# Workflow

Data Preprocessing : 
The dataset was scaled using StandardScaler.
Train-test split applied (X_train, X_test, y_train, y_test).

Model Training :
  Two models were trained:
    Logistic Regression
    Random Forest Classifier
Both models were fine-tuned using GridSearchCV.

Evaluation :
Models were evaluated using:
accuracy_score
confusion_matrix
classification_report

Deployment :
Streamlit app created for real-time prediction.

# Run Locally

Use Command : python -m streamlit run app.py
User inputs raw (unscaled) data → the app scales it with the same scaler used in training → model predicts.
