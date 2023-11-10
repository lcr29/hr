# HR Attrition Prediction App
# Overview
The HR Attrition Prediction app is a powerful tool designed to predict employee attrition based on various work-related and personal factors. This application is built using Streamlit and utilizes a machine learning model to provide predictions.

# Features
Predict the likelihood of employee attrition in an organization.
User-friendly interface with sidebar for easy input of relevant features.
Comprehensive input fields covering both personal and professional employee data.
Instant predictions on employee attrition risk.

# How to Use
Provide Employee Data: Use the sidebar to input employee details such as daily rate, distance from home, job role, education, and more.
Fill in Additional Information: Include categorical data like age group, business travel frequency, education field, and salary slab.
Submit for Prediction: After filling in the data, click the 'Predict' button to get the attrition risk prediction.

# Technical Details
The app loads a pre-trained hr_model.pkl for making predictions and a hr_preprocessor.pkl for data preprocessing.
Handles a mix of numerical and categorical inputs, ensuring they are properly encoded and formatted for the model.
Provides a binary prediction indicating either the likelihood of attrition ('Attrition') or stability ('No Attrition').

# Data Privacy
This app is designed for demonstration purposes and does not store user input data. Users should be mindful of the sensitivity of the data they input into the application.

# Disclaimer
This tool is intended for informational and demonstrative purposes only and should not be used as the sole basis for HR decisions.
