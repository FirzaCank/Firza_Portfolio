import streamlit as st
from PIL import Image
from EDA import eda
from ml import ml
from clean import clean
from PIL import Image
import pandas as pd

img = Image.open("Header\Header.png")
st.image(img)

df_load = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqlab_telco.csv')
df_final = pd.read_csv('dqlab_telco_final.csv')

def main():

    menu = ["Introduction","Data Cleansing","Exploratory Data Analysis","Machine Learning Model"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Introduction":
        st.title("Introduction")
        st.markdown("""**DQLab Telco** is a telecommunications subscription company services. 
        DQLab Telco is consistent in paying attention to its customer experience so
        that it will not be abandoned by customers.
        Despite the expectation, DQLab Telco already has many customers who switch
        subscriptions to competitors. The management wants to reduce the number of
        customers who churn by using **machine learning by creating the right
        prediction model to determine whether customers will unsubscribe (churn) or not.**""")

        st.caption("_The data is obtained from the dataset-dqlab csv format. The dataset link can be downloaded [here.](https://storage.googleapis.com/dqlab-dataset/dqlab_telco.csv)_")

        st.subheader("Customer Data of June 2020")
        st.caption("*This data is still in the form of raw data, which has not been carried out by the data cleansing and transformation process*")
        
        st.write("""In this data, the classification of churn and non-churn customers has been carried out.
        \nBasically, the classification of customer churn is seen from the definition made by the company itself.
        For example, customers are defined as churn when they have no longer transacted up to the last 6 months from the last available data update.""")
        st.write("the following is the raw dataset that will be used :")
        
        st.dataframe(df_load)

        st.text("""
        Columns Description :
        1. UpdatedAt         : Periode of Data taken
        2. customerID        : Customer ID
        3. gender            : Whether the customer is a male or a female (Male, Female)
        4. SeniorCitizen     : Whether the customer is a senior citizen or not (1, 0)
        5. Partner           : Whether the customer has a partner or not (Yes, No)
        6. Dependents        : Whether the customer has dependents or not (Yes, No)
        7. tenure            : Number of months the customer has stayed with the company
        8. PhoneService      : Whether the customer has a phone service or not (Yes, No)
        9. MultipleLines     : Whether the customer has multiple lines or not (Yes, No, No phone service)
        10. InternetService  : Customer’s internet service provider (DSL, Fiber optic, No)
        11. OnlineSecurity   : Whether the customer has online security or not (Yes, No, No internet service)
        12. OnlineBackup     : Whether the customer has online backup or not (Yes, No, No internet service)
        13. DeviceProtection : Whether the customer has device protection or not (Yes, No, No internet service)
        14. TechSupport      : Whether the customer has tech support or not (Yes, No, No internet service)
        15. StreamingTV      : Whether the customer has streaming TV or not (Yes, No, No internet service)
        16. StreamingMovies  : Whether the customer has streaming movies or not (Yes, No, No internet service)
        17. Contract         : The contract term of the customer (Month-to-month, One year, Two year)
        18. PaperlessBilling : Whether the customer has paperless billing or not (Yes, No)
        19. PaymentMethod    : The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
        20. MonthlyCharges   : The amount charged to the customer monthly
        21. TotalCharges     : The total amount charged to the customer
        22. Churn            : Whether the customer churned or not (Yes or No)
        """)
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("""The goal of this analysis is to build a machine learning model that can predict whether a customer will churn or not. Several steps will be taken, such as :
        \n1. Cleansing and Tranformation Data
        \n2. Exploratory Data Analysis (EDA)
        \n3. Build Machine Learning Model""")

    elif choice == "Data Cleansing":
        st.title("Data Cleansing")
        clean()

    elif choice == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis (EDA)")
        eda()

    else :
        ml()
    # else:
    #     st.title("About Me")
    #     socials = ["LinkedIn","GitHub","Gmail"]
    #     LinkedIn = "[Firza Chandra Sandjaya Putra](https://www.linkedin.com/in/firza-chandra-sandjaya-putra-246762136/)"
    #     Gmail = "[firzasandjaya@gmail.com](firzasandjaya@gmail.com)"
    #     GitHub = "[FirzaCank](https://github.com/FirzaCank/Firza_Portfolio)"
    #     linkedin = Image.open("E:\Portofolio\Logo socials\linkedin.png")
    #     st.image(linkedin, width=10)
    #     st.write("LinkedIn: ",LinkedIn)
    #     st.write("\nGitHub :",GitHub)
    #     st.write("\nGmail: ",Gmail)
main()

with st.sidebar:
    st.write(" ")
    st.text("Dataset by: ")
    imag = Image.open("dqlab.png")
    st.image(imag)

    st.write(" ")
    Linkedin = "[Firza Chandra Sandjaya Putra](https://www.linkedin.com/in/firza-chandra-sandjaya-putra-246762136/)"
    st.text("Streamlit by: ")
    st.write(Linkedin)