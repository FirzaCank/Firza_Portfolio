from sklearn.metrics import label_ranking_average_precision_score
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
pd.options.display.max_columns=50


def clean():
    df_load = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqlab_telco.csv')
    df_load = df_load[['UpdatedAt','customerID','gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV','InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn']]
    df_final = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqlab_telco_final.csv')

    st.write("""Before building the model, the data needs to be prepared by do data cleansing. Data cleansing is very import for making the analytics and machine learning models error-free.
    From the user's direction it says that it only requires a few columns to become a feature in machine learning, so that some columns that are not needed will be drop.""")
    st.write("The following are basic details of the data to check before data cleaning process :")
    check = ['New Raw Dataframe','Data Information','Data Description','Number of Unique Data']

    with st.expander("Click to see the details 👉"):
        choose = st.selectbox("Choose the option you want to view", check)
    
        if choose == 'New Raw Dataframe':
            st.dataframe(df_load)
        elif choose == 'Data Information':
            st.text("""<class 'pandas.core.frame.DataFrame'>
                    RangeIndex: 7113 entries, 0 to 7112
                    Data columns (total 13 columns):
                    #   Column            Non-Null Count  Dtype  
                    ---  ------            --------------  -----  
                    0   UpdatedAt         7113 non-null   int64  
                    1   customerID        7113 non-null   object 
                    2   gender            7113 non-null   object 
                    3   SeniorCitizen     7113 non-null   int64  
                    4   Partner           7113 non-null   object 
                    5   tenure            7014 non-null   float64
                    6   PhoneService      7113 non-null   object 
                    7   StreamingTV       7113 non-null   object 
                    8   InternetService   7113 non-null   object 
                    9   PaperlessBilling  7113 non-null   object 
                    10  MonthlyCharges    7087 non-null   float64
                    11  TotalCharges      7098 non-null   float64
                    12  Churn             7070 non-null   object 
                    dtypes: float64(3), int64(2), object(8)
                    memory usage: 722.5+ KB
                    None""")

        elif choose == 'Data Description':
            st.dataframe(df_load.describe())
        else:
            st.dataframe(df_load.nunique())
    
    st.subheader("1. Filtering Certain Format Customer ID Numbers")

    #Filtering the correct 'customerID' column
    df_load['valid_id'] = df_load['customerID'].astype(str).str.match(r'(45\d{9,10})')
    df_load = (df_load[df_load['valid_id'] == True]).drop('valid_id', axis=1)

    st.write("""Look for the correct CustomerID Customer ID Number (Phone Number) format, with the following criteria:
    \n- The character length is 11-12.
    \n- Consists of numbers only, no characters other than numbers are allowed
    \n- Starting with the number 45 the first 2 digits.""")

    with st.expander("Click to see the code 👉"):
        code = '''df_load['valid_id'] = df_load['customerID'].astype(str).str.match(r'(45\d{9,10})')
        df_load = (df_load[df_load['valid_id'] == True]).drop('valid_id', axis=1)
        print('The filtered result of the number of Customer IDs is',df_load['customerID'].count())'''
        st.code(code, language='python')

        st.text("#Output : The filtered result of the number of Customer IDs is 7006")
    st.write(" ")
    st.write(" ")

    st.subheader("2. Filtering Duplicate Customer ID Numbers")

    #Drop Duplicate Rows
    df_load.drop_duplicates()
    #Drop duplicate ID sorted by Periode
    df_load = df_load.sort_values('UpdatedAt', ascending=False).drop_duplicates(['customerID'])
    #Dropping null values on Churn column
    df_load.dropna(subset=['Churn'],inplace=True)
    
    st.write("""Ensure that there are no duplicate customer ID numbers. Usually this type of duplicate ID number:
    \n- Duplication due to inserting more than once with the same value for each column
    \n- Duplication due to inserting different data retrieval periods""")
    with st.expander("Click to see the code 👉"):
        st.code('''# Drop Duplicate Rows
        df_load.drop_duplicates()
        # Drop duplicate ID sorted by Periode
        df_load = df_load.sort_values('UpdatedAt', ascending=False).drop_duplicates(['customerID'])
        print('The result of the number of customer IDs that have been removed from the duplicate (distinct) is ',df_load['customerID'].count()) ''', language='python')
        st.text('#Output : The result of the number of customer IDs that have been removed from the duplicate (distinct) is  6993')
    st.write(" ")


    st.subheader("3. Overcoming Missing Values ​​by Deleting Rows")
    st.write("""Next we will delete rows from undetected data whether it churns or not. It is assumed that the data modeller will only accept data that has a churn flag or not.""")
    with st.expander("Click to see the code 👉"):
        st.code('''#Checking
        print('Total missing values data dari kolom Churn',df_load['Churn'].isnull().sum())
        # Dropping all Rows with spesific column (churn)
        df_load.dropna(subset=['Churn'],inplace=True)
        print('Total Rows and Data columns after deleting Missing Values data is',df_load.shape)''', language='python')
        st.text('''#Output : Total missing values data dari kolom Churn 43
        Total Rows and Data columns after deleting Missing Values data is (6950, 13)''')
    st.write(" ")
    st.write(" ")


    st.subheader("4. Overcoming Missing Values ​​by Filling in Certain Values (Imputation)")

    #handling missing values Tenure fill with 11 (according to directions)
    df_load['tenure'].fillna(11, inplace=True)
    #handling missing values num vars (except Tenure)
    for col_name in list(['MonthlyCharges','TotalCharges']):
        median = df_load[col_name].median()
        df_load[col_name].fillna(median, inplace=True)
    
    st.write("""The data modeler asks for missing values ​​to fill in with the following criteria:
    \n- The data modeller asks tenure column for each row that has missing values ​​for the length of the subscription to be filled with 11.
    \n- Variables that are numeric other than Tenure are filled with the median of each of these variables.""")
    with st.expander("Click to see the code 👉"):
        st.code('''print('Status Missing Values :',df_load.isnull().values.any())
        print('\nTotal Missing Values for each column, adalah:')
        print(df_load.isnull().sum().sort_values(ascending=False))

        # handling missing values Tenure fill with 11
        df_load['tenure'].fillna(11, inplace=True)

        # Handling missing values num vars (except Tenure)
        for col_name in list(['MonthlyCharges','TotalCharges']):
            median = df_load[col_name].median()
            df_load[col_name].fillna(median, inplace=True)
            
        print('\nThe number of Missing Values after imputering the data is:')
        print(df_load.isnull().sum().sort_values(ascending=False))''', language='python')
        st.text('''Status Missing Values : True

        Total Missing Values for each column, adalah:
        tenure              99
        MonthlyCharges      26
        TotalCharges        15
        UpdatedAt            0
        customerID           0
        gender               0
        SeniorCitizen        0
        Partner              0
        PhoneService         0
        StreamingTV          0
        InternetService      0
        PaperlessBilling     0
        Churn                0
        dtype: int64

        The number of Missing Values after imputering the data is:
        UpdatedAt           0
        customerID          0
        gender              0
        SeniorCitizen       0
        Partner             0
        tenure              0
        PhoneService        0
        StreamingTV         0
        InternetService     0
        PaperlessBilling    0
        MonthlyCharges      0
        TotalCharges        0
        Churn               0
        dtype: int64''')
    st.write(" ")
    st.write(" ")

    st.subheader('5. Detecting and Overcoming the Outliers')
    st.write('Detecting outliers can be seen with boxplot visualization and updated data description. These are the data description and plot **before overcoming the outliers** :')
    cek = ['Updated Data Description','Box Plot for Numerical Features','Code']
    with st.expander("Click to check the data description and boxplot 👉"):
        choose = st.selectbox("Choose the option you want to view", cek)
        if choose == 'Updated Data Description':
            st.write('Before Outliers Handling :')
            st.dataframe(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())
        elif choose == 'Box Plot for Numerical Features':
            # Creating Box Plot for num vars columns
            df_load = df_load[['UpdatedAt','customerID','gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV','InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn']]

            fig = plt.figure(figsize=(6,3))
            sns.boxplot(x=df_load['tenure'])
            st.pyplot(fig)

            fig = plt.figure(figsize=(6,3))
            sns.boxplot(x=df_load['MonthlyCharges'])
            st.pyplot(fig)

            fig = plt.figure(figsize=(6,3))
            sns.boxplot(x=df_load['TotalCharges'])
            st.pyplot(fig)
        
        else:
            st.code('''# Creating Box Plot for num vars columns
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure()
            sns.boxplot(x=df_load['tenure'])
            plt.show()

            plt.figure()
            sns.boxplot(x=df_load['MonthlyCharges'])
            plt.show()

            plt.figure()
            sns.boxplot(x=df_load['TotalCharges'])
            plt.show()''', language='python')

    st.write('After we know which variables have outliers, then we will overcome outliers by using the interquartile range (IQR) method. These are the data description and plot **after overcoming the outliers :**')
    
    #handling outliers with IQR
    Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
    Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)

    IQR = Q3 - Q1
    maximum  = Q3 + (1.5*IQR)
    minimum = Q1 - (1.5*IQR)

    more_than     = (df_load > maximum)
    lower_than    = (df_load < minimum)
    df_load       = df_load.mask(more_than, maximum, axis=1) 
    df_load       = df_load.mask(lower_than, minimum, axis=1)

    cek1 = ['Updated Data Description','Updated Box Plot for Numerical Features','Code']
    with st.expander("Click to check the Updated data description and boxplot 👉"):
        choose = st.selectbox("Choose the option you want to view", cek1)
        if choose == 'Updated Data Description':
            st.write('After Outliers Handling :')
            st.dataframe(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())
        elif choose == 'Updated Box Plot for Numerical Features':
            # Creating Box Plot for num vars columns
            df_load = df_load[['UpdatedAt','customerID','gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV','InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn']]

            fig = plt.figure(figsize=(6,3))
            sns.boxplot(x=df_load['tenure'])
            st.pyplot(fig)

            fig = plt.figure(figsize=(6,3))
            sns.boxplot(x=df_load['MonthlyCharges'])
            st.pyplot(fig)

            fig = plt.figure(figsize=(6,3))
            sns.boxplot(x=df_load['TotalCharges'])
            st.pyplot(fig)
        else:
            st.code('''# Handling with IQR
                Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
                Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)

                IQR = Q3 - Q1
                maximum  = Q3 + (1.5*IQR)
                print('The Maximum Value of each Variable is: ')
                print(maximum)
                minimum = Q1 - (1.5*IQR)
                print('\nThe Minimum Value of each Variable is: ')
                print(minimum)

                more_than     = (df_load > maximum)
                lower_than    = (df_load < minimum)
                df_load       = df_load.mask(more_than, maximum, axis=1) 
                df_load       = df_load.mask(lower_than, minimum, axis=1)''', language='python')
            st.text('''#Output : The Maximum Value of each Variable are: 
                    tenure             124.00000
                    MonthlyCharges     169.93125
                    TotalCharges      8889.13125
                    dtype: float64

                    The Minimum Value of each Variable are: 
                    tenure             -60.00000
                    MonthlyCharges     -43.61875
                    TotalCharges     -4682.31875
                    dtype: float64''')
    st.write(" ")
    st.write(" ")

    st.subheader('6. Detecting Non-Standard Values and Standardized it')
    st.write("Detects whether there are values ​​of non-standard categorical variables. This usually occurs due to data input errors. Differences in terms are one of the factors that often occur, for that we need standardization of the data that has been inputted.")
    
    with st.expander("Click to check the output and code 👉"):
        cek2 = ['Detecting Non-Standard Values','Code']
        choose2 = st.selectbox("Choose the option you want to view", cek2)
        if choose2 == 'Detecting Non-Standard Values':
            #Detecting Non-Standard Values
            for col_name in list(['gender','SeniorCitizen','Partner','PhoneService','StreamingTV','InternetService','PaperlessBilling','Churn']):
                st.write('Unique Values Count **Before Standardized** Variable', col_name)
                st.dataframe(df_load[col_name].value_counts())
                
                df_load['tenure'] = df_load['tenure'].astype(int)
        else:
            st.code('''\tfor col_name in list(['gender','SeniorCitizen','Partner','PhoneService','StreamingTV','InternetService','PaperlessBilling','Churn']):
    print('Unique Values Count **Before Standardized** Variable', col_name)
    print(df_load[col_name].value_counts())
            
df_load['tenure'] = df_load['tenure'].astype(int)''', language='python')

    st.write('After we know which variables have non-standard values, then we standardize them with the most patterns, provided that they do not change their meaning.')
    
    df_load = df_load.replace(['Wanita','Laki-Laki','Churn','Iya','No internet service','Fiber optic','DSL'],['Female','Male','Yes','Yes','No','Yes','Yes'])
    df_load['SeniorCitizen'].mask(df_load['SeniorCitizen'] == 0, 'No', inplace=True)
    df_load['SeniorCitizen'].mask(df_load['SeniorCitizen'] == 1, 'Yes', inplace=True)
     
    with st.expander("Click to check the output and code 👉"):
        cek3 = ['Check unique values after standardized','Code']
        choose3 = st.selectbox("Choose the option you want to view", cek3)
        if choose3 == 'Check unique values after standardized':
            #Check unique values after standardized
            for col_name in list(['gender','SeniorCitizen','Partner','PhoneService','StreamingTV','InternetService','PaperlessBilling','Churn']):
                st.write('Unique Values Count **After Standardized** Variable',col_name)
                st.dataframe(df_load[col_name].value_counts())
        else:
            st.code('''df_load = df_load.replace(['Wanita','Laki-Laki','Churn','Iya','No internet service','Fiber optic','DSL'],['Female','Male','Yes','Yes','No','Yes','Yes'])
            df_load['SeniorCitizen'].mask(df_load['SeniorCitizen'] == 0, 'No', inplace=True)
            df_load['SeniorCitizen'].mask(df_load['SeniorCitizen'] == 1, 'Yes', inplace=True)
            
            for col_name in list(['gender','SeniorCitizen','Partner','PhoneService','StreamingTV','InternetService','PaperlessBilling','Churn']):
                print('Unique Values Count **After Standardized** Variable',col_name)
                print(df_load[col_name].value_counts())''',language='python')
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.subheader("The latest data that has passed the data cleansing process")
    
   
    st.dataframe(df_final)