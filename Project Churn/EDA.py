import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def eda():
    df = pd.read_csv('dqlab_telco_final.csv')
    df = df.drop(['customerID','UpdatedAt'], axis=1)
    num = ['MonthlyCharges','TotalCharges','tenure']
    cat = ['Gender','SeniorCitizen','Partner','PhoneService','StreamingTV','InternetService','PaperlessBilling']

    st.write("Exploration Data Analysis allows analysts to understand the content of the data used, from distribution, frequency, correlation and more.")

    #Visualize churn percentage
    st.subheader("Churn Percentage Visualization")
    labels = ['Yes','No'] #churn value
    churn = df.Churn.value_counts()
    fig = px.pie(values = churn, names=labels) #autocpt for display percetage
    st.plotly_chart(fig)

    st.write(" It can be seen that distribution of data as a whole or most customers do not churn, with details of **Churn as much as 26.4%** and **No Churn as much as 73.6%.**")
    st.write(" ")
    st.write(" ")

    #Numerical Features Plot
    st.header("Customer Churn Plot based on Numerical Variable")
    st.write("The next step is to choose a numeric predictor variable and make a bivariate plot.")
    # choice = st.sidebar.selectbox("Exploratory Data Analysis - Numerical Features", num)

    with st.expander("Numerical Variable"):
            choose = st.selectbox("Choose the plot you want to view", num)

            if choose == "MonthlyCharges":
                st.subheader("Customer Churn Plot of Monthly Charges Visualization")
                fig, ax = plt.subplots(1, 1)
                #plot two overlays of histogram per each numerical_features, use a color of blue and orange
                df[df.Churn == 'No']['MonthlyCharges'].hist(bins=20, color='blue', alpha=0.5, ax=ax)
                df[df.Churn == 'Yes']['MonthlyCharges'].hist(bins=20, color='orange', alpha=0.5, ax=ax)
                plt.legend(['No','Yes'])
                plt.xlabel("Monthly Charges")
                plt.ylabel("Churn / No Churn")
                st.pyplot(fig)

            elif choose == "TotalCharges":
                st.subheader("Customer Churn Plot of Total Charges Visualization")
                fig, ax = plt.subplots(1, 1)
                #plot two overlays of histogram per each numerical_features, use a color of blue and orange
                df[df.Churn == 'No']['TotalCharges'].hist(bins=20, color='blue', alpha=0.5, ax=ax)
                df[df.Churn == 'Yes']['TotalCharges'].hist(bins=20, color='orange', alpha=0.5, ax=ax)
                plt.legend(['No','Yes'])
                plt.xlabel("Total Charges")
                plt.ylabel("Churn / No Churn")
                st.pyplot(fig)

            else:
                st.subheader("Customer Churn Plot of Tenure Visualization")
                fig, ax = plt.subplots(1, 1)
                #plot two overlays of histogram per each numerical_features, use a color of blue and orange
                df[df.Churn == 'No']['tenure'].hist(bins=20, color='blue', alpha=0.5, ax=ax)
                df[df.Churn == 'Yes']['tenure'].hist(bins=20, color='orange', alpha=0.5, ax=ax)
                plt.legend(['No','Yes'])
                plt.xlabel("Tenure")
                plt.ylabel("Churn / No Churn")
                st.pyplot(fig)

    st.write("""- It is known that for **MonthlyCharges** there is a tendency for the smaller the monthly fee value to be charged, the smaller tendency to Churn.
    \n- For **TotalCharges**, there seems to be no trend towards Churn customers.
    \n- For **tenure**, there is a tendency that the longer the customer subscribes, the smaller the tendency to Churn.""")
    st.write(" ")
    st.write(" ")

    #Cateogrical Features Plot
    sns.set(style='darkgrid')
    st.header("Customer Churn Plot based on Categorical Variable")
    st.write("The next step is to choose a categorical predictor variable and make a bivariate plot.")


    #choice1 = st.sidebar.selectbox("Exploratory Data Analysis - Categorical Features", cat)

    with st.expander("Categorical Variable"):
            choose = st.selectbox("Choose the plot you want to view", cat)

            if choose == "Gender":
                st.subheader("Customer Churn Plot of Gender Visualization")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='gender', hue='Churn')
                plt.tight_layout()
                st.pyplot(fig)
                
            elif choose == "Partner":
                st.subheader("Customer Churn Plot of Partner Visualization")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='Partner', hue='Churn')
                plt.tight_layout()
                st.pyplot(fig)
                
            elif choose == "SeniorCitizen":
                st.subheader("Customer Churn Plot of SeniorCitizen Visualization")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='SeniorCitizen', hue='Churn')
                plt.tight_layout()
                st.pyplot(fig)

            elif choose == "PhoneService":
                st.subheader("Customer Churn Plot of PhoneService Visualization")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='PhoneService', hue='Churn')
                plt.tight_layout()
                st.pyplot(fig)
                
            elif choose == "StreamingTV":
                st.subheader("Customer Churn Plot of StreamingTV Visualization")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='StreamingTV', hue='Churn')
                plt.tight_layout()
                st.pyplot(fig)
                
            elif choose == "InternetService":
                st.subheader("Customer Churn Plot of InternetService Visualization")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='InternetService', hue='Churn')
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.subheader("Customer Churn Plot of PaperlessBilling Visualization")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='PaperlessBilling', hue='Churn')
                plt.tight_layout()
                st.pyplot(fig)
        
    st.write("""- It is known that there is no significant difference for customer who do churn in terms of gender **(gender)** and telephone service **(PhoneService)**.
\nHowever, there is a **_tendency that customer who churn_** are :
\n- Customer who do not have partners **(partners: No)**
\n- Customer whose status is senior citizens **(SeniorCitizen: Yes)**
\n- Customer who have TV streaming services **(StreamingTV: Yes)**
\n- Customer who have Internet service **(internetService: Yes)**
\n- Customer whose bills are paperless **PaperlessBilling: Yes)**.""")

    #col1, col2 = st.columns(2)
    #with col1: 
        #with st.expander("gender"):
            #st.dataframe(df['gender'].value_counts())
    