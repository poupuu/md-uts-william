import streamlit as st
import numpy as np
import joblib
import pickle
from xgboost import XGBClassifier

# Load model
model = joblib.load("AllPickle/XGBModel.pkl")

# Load encoders
genderEncoder = joblib.load("AllPickle/genderEncoder.pkl")
educationEncoder = joblib.load("AllPickle/educationEncoder.pkl")
homeOwnershipEncoder = joblib.load("AllPickle/homeOwnershipEncoder.pkl")
loanIntentEncoder = joblib.load("AllPickle/loanIntentEncoder.pkl")
previousFileEncoder = joblib.load("AllPickle/previousFileEncoder.pkl")

# Load scalers
ageScaler = joblib.load("AllPickle/ageScaler.pkl")
incomeScaler = joblib.load("AllPickle/incomeScaler.pkl")
empExpScaler = joblib.load("AllPickle/empExpScaler.pkl")
loanAmountScaler = joblib.load("AllPickle/loanAmountScaler.pkl")
intRateScaler = joblib.load("AllPickle/intRateScaler.pkl")
loanPercentScaler = joblib.load("AllPickle/loanPercentScaler.pkl")
personCredScaler = joblib.load("AllPickle/personCredScaler.pkl")
creditScoreScaler = joblib.load("AllPickle/creditScoreScaler.pkl")

def split_data(data, target_column="loan_status"):
  output_df = data[target_column]
  input_df = data.drop(target_column, axis=1)
  return input_df, output_df

def convert_input_to_df(input_data):
  data = [input_data]
  df = pd.DataFrame(data, columns = ['person_age', 'person_gender', 'person_education', 'person_income', 'person_emp_exp', 'person_home_ownership', 'loan_amnt ', 'loan_intent', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file'])
  return df

def feature_encode(df):
  df["person_gender"] = genderEncoder.transform(df[["person_gender"]])
  df["person_education"] = educationEncoder.transform(df[["person_education"]])
  df["person_home_ownership"] = homeOwnershipEncoder.transform(df[["person_home_ownership"]])
  df["loan_intent"] = loanIntentEncoder.transform(df[["loan_intent"]])
  df["previous_loan_defaults_on_file"] = previousFileEncoder.transform(df[["previous_loan_defaults_on_file"]])
  return df

def feature_scaling(df):
  df["person_age"] = ageScaler.transform(df[["person_age"]])
  df["person_income"] = incomeScaler.transform(df[["person_income"]])
  df["person_emp_exp"] = empExpScaler.transform(df[["person_emp_exp"]])
  df["loan_amnt"] = loanAmountScaler.transform(df[["loan_amnt"]])
  df["loan_int_rate"] = intRateScaler.transform(df[["loan_int_rate"]])
  df["loan_percent_income"] = loanPercentScaler.transform(df[["loan_percent_income"]])
  df["cb_person_cred_hist_length"] = personCredScaler.transform(df[["cb_person_cred_hist_length"]])
  df["credit_score"] = creditScoreScaler.transform(df[["credit_score"]])
  return df

def predictionLoan(user_input):
  prediction = model.predict(user_input)
  decoded_prediction = targetEncoder.inverse_transform([[prediction[0]]])  
  return decoded_prediction[0][0]

def classification_proba(user_input):
  predictProba = model.predict_proba(user_input)
  column_names = [
        "Denied", "Approved"
  ]
  probaDF = pd.DataFrame(predictProba, columns=column_names)
  return probaDF

# UI
st.title("Loan Approval Prediction")

# Input fields
st.subheader("Input Loan Data")
person_gender = st.selectbox("person_gender", ("Male", "Female"))
person_education = st.selectbox("person_education", ("High School", "Associate", "Bachelor", "Master", "Doctorate"))
person_home_ownership = st.selectbox("person_home_ownership", ("OTHER", "RENT", "MORTGAGE", "OWN"))
loan_intent = st.selectbox("loan_intent", ("MEDICAL", "EDUCATION", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "PERSONAL", "VENTURE"))
previous_loan_defaults_on_file = st.selectbox("previous_loan_defaults_on_file", ("No", "Yes"))

person_age = st.slider('person_age', min_value=21, max_value=70, value=21)
person_income = st.number_input('person_income', value=2.448661e+06)
person_emp_exp = st.slider('person_emp_exp', min_value=0, max_value=40, value=5)
loan_amnt = st.number_input('loan_amnt', value=10000.0)
loan_int_rate = st.slider('loan_int_rate', min_value=0.0, max_value=30.0, value=12.5)
loan_percent_income = st.slider('loan_percent_income', min_value=0.0, max_value=100.0, value=10.0)
cb_person_cred_hist_length = st.slider('cb_person_cred_hist_length', min_value=0.0, max_value=50.0, value=10.0)
credit_score = st.slider('credit_score', mien_value=300, max_value=850, value=600)

input_data = [person_age, person_gender, person_education, person_income, person_emp_exp, person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file]

user_df = convert_input_to_df(input_data)

st.subheader("Inputted Patient Data")
st.dataframe(user_df)

user_df = feature_encode(user_df)
user_df = feature_scaling(user_df)

prediction = predictionLoan(user_df)
proba = classification_proba(user_df)

st.subheader("Prediction Result")
st.dataframe(proba)
st.write('The predicted output is: ', prediction)
  
if __name__ == "__main__":
  main()

# # Predict
# if st.button("Predict Loan Approval"):
#     prediction = model.predict(user_input)
#     result = "Approved" if prediction[0] == 1 else "Not Approved"
#     st.success(f"Loan Status Prediction: **{result}**")
