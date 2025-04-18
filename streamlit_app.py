# import streamlit as st
# import joblib
# import pandas as pd

# genderEncoder = joblib.load("AllPickle/genderEncoder.pkl")
# educationEncoder = joblib.load("AllPickle/educationEncoder.pkl")
# homeOwnershipEncoder = joblib.load("AllPickle/homeOwnershipEncoder.pkl")
# loanIntentEncoder = joblib.load("AllPickle/loanIntentEncoder.pkl")
# previousFileEncoder = joblib.load("AllPickle/previousFileEncoder.pkl")

# ageScaler = joblib.load("AllPickle/ageScaler.pkl")
# incomeScaler = joblib.load("AllPickle/incomeScaler.pkl")
# empExpScaler = joblib.load("AllPickle/empExpScaler.pkl")
# loanAmountScaler = joblib.load("AllPickle/loanAmountScaler.pkl")
# intRateScaler = joblib.load("AllPickle/intRateScaler.pkl")
# loanPercentScaler = joblib.load("AllPickle/loanPercentScaler.pkl")
# personCredScaler = joblib.load("AllPickle/personCredScaler.pkl")
# creditScoreScaler = joblib.load("AllPickle/creditScoreScaler.pkl")

# def split_data(data, target_column="loan_status"):
#   output_df = data[target_column]
#   input_df = data.drop(target_column, axis=1)
#   return input_df, output_df

# def feature_encode(df):
#   df["person_gender"] = genderEncoder.transform(df[["person_gender"]])
#   df["person_education"] = educationEncoder.transform(df[["person_education"]])
#   df["person_home_ownership"] = homeOwnershipEncoder.transform(df[["person_home_ownership"]])
#   df["loan_intent"] = loanIntentEncoder.transform(df[["loan_intent"]])
#   df["previous_loan_defaults_on_file"] = previousFileEncoder.transform(df[["previous_loan_defaults_on_file"]])
#   return df

# def predictionLoan(user_input):
#   prediction = model.predict(user_input)
#   decoded_prediction = targetEncoder.inverse_transform([[prediction[0]]])  
#   return decoded_prediction[0][0]

# def classification_proba(user_input):
#   predictProba = model.predict_proba(user_input)
#   column_names = [
#         "Denied", "Approved"
#   ]
#   probaDF = pd.DataFrame(predictProba, columns=column_names)
#   return probaDF

# def main():
#   st.title('Loan Classification')
#   st.info("This app use machine learning to classify loan approved or not.")

#   st.subheader("Dataset")
#   df = pd.read_csv("Dataset_A_loan.csv")
#   x, y = split_data(df)
  
#   with st.expander("**Raw Data**"):
#     st.dataframe(df.head(50))

#   with st.expander("**Input Data**"):
#     st.dataframe(x.head(50))

#   with st.expander("**Output Data**"):
#     st.dataframe(y.head(50))

#   st.subheader("Input Loan Data")
#   Age = st.slider('person_age', min_value = 21, max_value = 70, value = 21)
#   Income = st.slider('person_income', min_value = 8000, max_value = 2.00, value = 1.75)
#   Weight = st.slider('Weight', min_value = 30, max_value = 180, value = 70)
#   FCVC = st.slider('FCVC', min_value = 1, max_value = 3, value = 2)
#   NCP = st.slider('NCP', min_value = 1, max_value = 4, value = 3)
#   CH2O = st.slider('CH2O', min_value = 1, max_value = 3, value = 2)
#   FAF = st.slider('FAF', min_value = 0, max_value = 3, value = 1)
#   TUE = st.slider('TUE', min_value = 0, max_value = 2, value = 1)
  
#   Gender = st.selectbox('Gender', ('Male', 'Female'))
#   family_history_with_overweight = st.selectbox('Family history with overweight', ('yes', 'no'))
#   FAVC = st.selectbox('FAVC', ('yes', 'no'))
#   CAEC = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'no'))
#   SMOKE = st.selectbox('SMOKE', ('yes', 'no'))
#   SCC = st.selectbox('SCC', ('yes', 'no'))
#   CALC = st.selectbox('CALC', ('Sometimes', 'no', 'Frequently', 'Always'))
#   MTRANS = st.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))

#   input_data = [Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]

#   user_df = convert_input_to_df(input_data)

#   st.subheader("Inputted Patient Data")
#   st.dataframe(user_df)

#   user_df = encode_features(user_df)
#   user_df = normalize_features(user_df)

#   prediction = predict_classification(user_df)
#   proba = classification_proba(user_df)

#   st.subheader("Prediction Result")
#   st.dataframe(proba)
#   st.write('The predicted output is: ', prediction)
  

# if __name__ == "__main__":
#   main()

import streamlit as st
import numpy as np
import joblib
import pickle
from xgboost import XGBClassifier

# Load model
# with open("AllPickle/XGBModel.pkl", "rb") as file:
#     models = pickle.load(file)
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

# UI
st.title("Loan Approval Prediction")

# Input fields
st.subheader("Input Loan Data")
gender = st.selectbox("person_gender", ("Male", "Female"))
education = st.selectbox("person_education", ("High School", "Associate", "Bachelor", "Master", "Doctorate"))
home_ownership = st.selectbox("person_home_ownership", ("OTHER", "RENT", "MORTGAGE", "OWN"))
loan_intent = st.selectbox("loan_intent", ("MEDICAL", "EDUCATION", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "PERSONAL", "VENTURE"))
previous_default = st.selectbox("previous_loan_defaults_on_file", ("No", "Yes"))

age = st.slider('person_age', min_value=21, max_value=70, value=21)
income = st.number_input('Aperson_income', value=2.448661e+06)
emp_length = st.slider('person_emp_exp', min_value=0, max_value=40, value=5)
loan_amount = st.number_input('loan_amnt', value=10000.0)
int_rate = st.slider('loan_int_rate', min_value=0.0, max_value=30.0, value=12.5)
loan_percent_income = st.slider('loan_percent_income', min_value=0.0, max_value=100.0, value=10.0)
person_cred = st.slider('cb_person_cred_hist_length', min_value=0.0, max_value=50.0, value=10.0)
credit_score = st.slider('credit_score', min_value=300, max_value=850, value=600)

# Preprocessing
gender_encoded = genderEncoder.transform([gender])[0]
education_encoded = educationEncoder.transform([education])[0]
home_encoded = homeOwnershipEncoder.transform([home_ownership])[0]
loan_intent_encoded = loanIntentEncoder.transform([loan_intent])[0]
prev_default_encoded = previousFileEncoder.transform([previous_default])[0]

scaled_age = ageScaler.transform([[age]])[0][0]
scaled_income = incomeScaler.transform([[income]])[0][0]
scaled_emp = empExpScaler.transform([[emp_length]])[0][0]
scaled_loan = loanAmountScaler.transform([[loan_amount]])[0][0]
scaled_int_rate = intRateScaler.transform([[int_rate]])[0][0]
scaled_loan_percent = loanPercentScaler.transform([[loan_percent_income]])[0][0]
scaled_person_cred = personCredScaler.transform([[person_cred]])[0][0]
scaled_credit_score = creditScoreScaler.transform([[credit_score]])[0][0]

# Combine input
user_input = np.array([[scaled_age, scaled_income, gender_encoded, education_encoded,
                        home_encoded, loan_intent_encoded, scaled_emp,
                        scaled_loan, scaled_int_rate, scaled_loan_percent,
                        scaled_person_cred, scaled_credit_score, prev_default_encoded]])

# Predict
if st.button("Predict Loan Approval"):
    prediction = model.predict(user_input)
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    st.success(f"Loan Status Prediction: **{result}**")
