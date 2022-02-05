import pickle
import streamlit as st

# loading the trained model

pickle_in = open('loan.pkl', 'rb')
classifier = pickle.load(pickle_in)

@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction( Gender, Married, Education, Self_Employed, Property_Area, Loan_Status, Dependents, ApplicantIncomeLog, LoanAmountLog, Loan_Amount_Term_Log, Total_Income_Log):

    # Pre-processing user input
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0

    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1

    if Education == "Graduate":
        Education = 0
    else:
        Education = 1

    if  Self_Employed =="NO":
        Self_Employed = 0
    else:
        Self_Employed = 1

    if Property_Area =="Urban":
        Property_Area = 2
    else:
        Property_Area = 0

    if Loan_Status =="Y":
        Loan_Status = 1
    else:
        Loan_Status = 0

    if Dependents =="0":
        Dependents = 0
    else:
        Dependents = 1



    # Making predictions
    prediction = classifier.predict([[Gender, Married, Education, Self_Employed, Property_Area, Loan_Status, Dependents, ApplicantIncomeLog, LoanAmountLog,	Loan_Amount_Term_Log, Total_Income_Log]])

    if prediction == 1:
        pred = 'Approved'
    else:
        pred = 'Rejected'
    return pred


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:pink;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married"))
    Education = st.selectbox(' Education ', ("Graduate", "not Graduate"))
    Self_Employed = st.selectbox(' Self_Employed ', ("yes", "no"))
    Property_Area = st.selectbox('Property_Area  ', ("urban", "rural"))
    Credit_History = st.selectbox('Credit_History', ("Clear Debts", "Unclear Debts"))
    Dependents = st.selectbox('Dependents', ("0", "1"))
    Loan_Status = st.selectbox(' Loan_Status', ("Y", "N"))
    ApplicantIncomeLog= st.number_input("Enter ApplicantIncomeLog")
    LoanAmountLog = st.number_input("Enter LoanAmountLog")
    Loan_Amount_Term_Log = st.number_input("Enter Loan_Amount_Term_Log")
    Total_Income_Log = st.number_input("Enter Total_Income_Log")

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(Gender, Married, Education, Self_Employed, Property_Area, Loan_Status, Dependents,ApplicantIncomeLog, LoanAmountLog, Loan_Amount_Term_Log, Total_Income_Log)

        st.success('THE STATUS OF LOAN IS {}'.format(result))
        print(Loan_Status)

if __name__ == '__main__':
     main()






