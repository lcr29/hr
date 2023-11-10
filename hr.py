import streamlit as st
import pandas as pd
import joblib

# Load the model and preprocessor
model = joblib.load('hr_model.pkl')
preprocessor = joblib.load('hr_preprocessor.pkl')

# Streamlit webpage title
st.image('header_image.png', width=150) 
st.title("HR Attrition Prediction")

# Sidebar for user input
st.sidebar.header('User Input Features')

# Function to get user input
def user_input_features():
    # Numerical inputs
    daily_rate = st.sidebar.number_input("Daily Rate", min_value=0)
    distance_from_home = st.sidebar.number_input("Distance From Home", min_value=0)
    education = st.sidebar.number_input("Education", min_value=1, max_value=5)
    environment_satisfaction = st.sidebar.number_input("Environment Satisfaction", min_value=1, max_value=4)
    hourly_rate = st.sidebar.number_input("Hourly Rate", min_value=0)
    job_involvement = st.sidebar.number_input("Job Involvement", min_value=1, max_value=4)
    job_level = st.sidebar.number_input("Job Level", min_value=1, max_value=5)
    job_satisfaction = st.sidebar.number_input("Job Satisfaction", min_value=1, max_value=4)
    monthly_rate = st.sidebar.number_input("Monthly Rate", min_value=0)
    num_companies_worked = st.sidebar.number_input("Num Companies Worked", min_value=0)
    performance_rating = st.sidebar.number_input("Performance Rating", min_value=1, max_value=4)
    relationship_satisfaction = st.sidebar.number_input("Relationship Satisfaction", min_value=1, max_value=4)
    stock_option_level = st.sidebar.number_input("Stock Option Level", min_value=0)
    total_working_years = st.sidebar.number_input("Total Working Years", min_value=0)
    training_times_last_year = st.sidebar.number_input("Training Times Last Year", min_value=0)
    work_life_balance = st.sidebar.number_input("Work Life Balance", min_value=1, max_value=4)
    years_at_company = st.sidebar.number_input("Years At Company", min_value=0)
    years_in_current_role = st.sidebar.number_input("Years In Current Role", min_value=0)
    years_since_last_promotion = st.sidebar.number_input("Years Since Last Promotion", min_value=0)
    years_with_curr_manager = st.sidebar.number_input("Years With Curr Manager", min_value=0)

    # Categorical inputs
    age_group = st.sidebar.selectbox("Age Group", ['18-25', '26-35', '36-45', '46-55', '55+'])
    business_travel = st.sidebar.selectbox("Business Travel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'TravelRarely'])
    education_field = st.sidebar.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    job_role = st.sidebar.selectbox("Job Role", ['Laboratory Technician', 'Sales Representative', 'Research Scientist', 'Human Resources', 'Manufacturing Director', 'Sales Executive', 'Healthcare Representative', 'Research Director', 'Manager'])
    marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Divorced', 'Married'])
    salary_slab = st.sidebar.selectbox("Salary Slab", ['Upto 5k', '5k-10k', '10k-15k', '15k+'])
    over_time = st.sidebar.selectbox("Over Time", ['No', 'Yes'])

    # Organize user input into a DataFrame
    data = {
        'DailyRate': daily_rate, 
        'DistanceFromHome': distance_from_home, 
        'Education': education, 
        'EnvironmentSatisfaction': environment_satisfaction, 
        'HourlyRate': hourly_rate, 
        'JobInvolvement': job_involvement,
        'JobLevel': job_level, 
        'JobSatisfaction': job_satisfaction, 
        'MonthlyRate': monthly_rate, 
        'NumCompaniesWorked': num_companies_worked,
        'PerformanceRating': performance_rating, 
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level, 
        'TotalWorkingYears': total_working_years, 
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance, 
        'YearsAtCompany': years_at_company, 
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion, 
        'YearsWithCurrManager': years_with_curr_manager,
        'AgeGroup': age_group, 
        'BusinessTravel': business_travel, 
        'EducationField': education_field,
        'Gender': gender, 
        'JobRole': job_role, 
        'MaritalStatus': marital_status,
        'SalarySlab': salary_slab, 
        'OverTime': over_time
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the input data
st.subheader('User Input features')
st.write(input_df)

# Preprocess the input data
input_data_processed = preprocessor.transform(input_df)

# Predict and display the output
if st.button('Predict'):
    prediction = model.predict(input_data_processed)
    st.subheader('Prediction')
    st.write('Attrition' if prediction[0] == 1 else 'No Attrition')
