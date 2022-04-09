#import library
import pandas as pd
pd.options.display.max_columns=50
#import dataset
df_load = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqlab_telco.csv')

#Filtering the correct 'customerID' column
df_load['valid_id'] = df_load['customerID'].astype(str).str.match(r'(45\d{9,10})')
df_load = (df_load[df_load['valid_id'] == True]).drop('valid_id', axis=1)

#Drop Duplicate Rows
df_load.drop_duplicates()
#Drop duplicate ID sorted by Periode
df_load = df_load.sort_values('UpdatedAt', ascending=False).drop_duplicates(['customerID'])
#Dropping null values on Churn column
df_load.dropna(subset=['Churn'],inplace=True)

#handling missing values Tenure fill with 11 (according to directions)
df_load['tenure'].fillna(11, inplace=True)
#handling missing values num vars (except Tenure)
for col_name in list(['MonthlyCharges','TotalCharges']):
    median = df_load[col_name].median()
    df_load[col_name].fillna(median, inplace=True)

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

#Detecting Non-Standard Values
for col_name in list(['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']):
    print('\nUnique Values Count \033[1m' + 'Before Standardized \033[0m Variable',col_name)
    print(df_load[col_name].value_counts())

df_load['tenure'] = df_load['tenure'].astype(int)
    
print('------------------------------------------------------------------------------------------------------------------------')    

df_load = df_load.replace(['Wanita','Laki-Laki','Churn','Iya','No internet service','Fiber optic','DSL'],['Female','Male','Yes','Yes','No','Yes','Yes'])
df_load['SeniorCitizen'].mask(df_load['SeniorCitizen'] == 0, 'No', inplace=True)
df_load['SeniorCitizen'].mask(df_load['SeniorCitizen'] == 1, 'Yes', inplace=True)

#Check unique values after standardized
for col_name in list(['gender','Dependents','Churn','SeniorCitizen']):
    print('\nUnique Values Count \033[1m' + 'After Standardized \033[0mVariable',col_name)
    print(df_load[col_name].value_counts())

df = df_load[['UpdatedAt','customerID','gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV','InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn']]
