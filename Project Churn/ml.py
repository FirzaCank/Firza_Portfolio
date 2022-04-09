from base64 import encode
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from pathlib import Path
from PIL import Image
import os
import joblib


def ml():
    df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqlab_telco_final.csv')

    #Remove the unnecessary columns customerID & UpdatedAt
    df = df.drop(['customerID','UpdatedAt'], axis=1)

    #Convert all the non-numeric columns to numerical data types
    for column in df.columns:
        if df[column].dtype == np.number: continue
        # Perform encoding for each non-numeric column
        df[column] = LabelEncoder().fit_transform(df[column])

    #Predictor and target
    X = df.drop('Churn', axis = 1) 
    y = df['Churn']
    #Splitting train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    submenu = ["Data Preprocessing","Classification Report","Confusion Matrix","Prediction",'Conclusion']
    choice = st.sidebar.selectbox("Machine Learning Menu", submenu)
    split = ['Data Train','Data Test']
    model = ['Training Model','Testing Model']


    if choice == "Data Preprocessing":
        st.title("Machine Learning Model - Data Preprocessing")
        st.write("At this preprocessing stage, **unnecessary column removal, data encoding, and splitting dataset** will be carried out so that the data is ready for modeling purposes.")
        st.write(" ")
        st.write(" ")
        st.subheader("Removing Unnescessary Columns")
        st.write("There are columns that are not needed for modeling, namely the **_'UpdatedAt'_** and **_'customerID'_** columns, so they will be deleted.")
        with st.expander("Click to see the code 👉"):
            st.code("cleaned_df = df_load.drop(['customerID','UpdatedAt'], axis=1)", language='python')
        st.write(" ")
        st.write(" ")

        
        st.subheader("Data Encoding")
        st.write("The purpose of this step is to change the value of the data that is still in the form of a string to be converted into numeric form")
        st.write("This is **the updated dataset after data encoding** :")
        st.dataframe(df)

        with st.expander("Click to see the data description after encoding 👉"):
            s = ['Data Description after Encoding','Code']
            choose = st.selectbox("Choose the option you want to view", s)
            if choose == 'Data Description after Encoding':
                st.dataframe(df.describe())
            else:
                st.code("""#Convert all the non-numeric columns to numerical data types
for column in cleaned_df.columns:
    if cleaned_df[column].dtype == np.number: continue
    # Perform encoding for each non-numeric column
    cleaned_df[column] = LabelEncoder().fit_transform(cleaned_df[column])
""", language='python')
        st.write("it can be seen that the distribution of the data, especially **the min and max columns of each categorical variable, has changed to 0 & 1.**")
        st.write(" ")
        st.write(" ")

        st.subheader("Splitting Dataset")
        st.write("in this step will divide the dataset into 2 parts (70% training & 30% testing) based on the predictor variable (X) and the target (Y). In this output below is output to check that the distribution of the dataset is in accordance with the proportions")
        # Print according to the expected result
        st.write('The number of rows and columns of data train of predictor variable (X) is:', x_train.shape,
        '\n, while the number of rows and columns of data train of target variable (Y) is:', y_train.shape,
        '\nThe churn percentage in data training is:',
        '\n',y_train.value_counts(normalize=True),
        '\nThe number of rows and columns of data test of predictor variable (X) is:', x_test.shape,',\nwhile the number of rows and columns of data test of target variable (Y) is:', y_test.shape,
        '\nThe churn percentage in data testing is:',
        '\n',y_test.value_counts(normalize=True))
        with st.expander("Click to see the code 👉"):
            st.code("""from sklearn.model_selection import train_test_split
# Predictor dan target
X = cleaned_df.drop('Churn', axis = 1) 
y = cleaned_df['Churn']
# Splitting train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Print according to the expected result
print('The number of rows and columns of data train of predictor variable (X) is:', x_train.shape,', while the number of rows and columns of data train of target variable (Y) is:', y_train.shape)
print('The churn percentage in data training is:')
print(y_train.value_counts(normalize=True))
print('The number of rows and columns of data test of predictor variable (X) is:', x_test.shape,', while the number of rows and columns of data test of target variable (Y) is:', y_test.shape)
print('The churn percentage in data testing is:')
print(y_test.value_counts(normalize=True))
""", language='python')
        st.write("After that, it can be seen that the number of rows and columns of each data is appropriate & the percentage of the churn column is also the same as the data at the beginning, this indicates that the **data is separated properly and correctly.**")

    elif choice == "Classification Report":
        st.title("Machine Learning Model - Classification Report")
        st.write("The machine learning model that will be built uses three algorithms, including **Logistic Regression, Random Forest and Gradiant Boost Classifier** by using the default without adding any parameters. Then the best model with the highest accuracy will be selected.")
        st.write("In this section, a classification report will be presented for each model algorithm.")
        st.subheader("Classification Report of Logistic Regression")
        # Print classification report 
        with st.expander("Data Train or Data Test"):
                choose = st.selectbox("View the Classification Report of Logistic Regression Model for :", split)

                if choose == "Data Train":
                    st.subheader("The Classification Report Training Model of Logistic Regression")
                    st.write("From the **training data**, it can be seen that the model is able to predict the data by producing an **accuracy of 80%.**")
                    img = Image.open("E:\Portofolio\classification report image\log train.jpg")
                    st.image(img)
                
                else:
                    st.subheader("The Classification Report Testing Model of Logistic Regression")
                    st.write(" Meanwhile From the **data testing**, it can be seen that the model is able to predict the data by producing an **accuracy of 79%.**")
                    img = Image.open("E:\Portofolio\classification report image\log test.jpg")
                    st.image(img)

        st.write(" ")
        st.write(" ")
        # Print classification report
        st.subheader("Classification Report of Random Forest Classifier")
        with st.expander("Data Train or Data Test"):
                choose = st.selectbox("View the Classification Report of Random Forest Classifier Model for :", split)

                if choose == "Data Train":
                    st.subheader("The Classification Report Training Model of Random Forest Classifier")
                    st.write("From the **training data**, it can be seen that the model is able to predict the data by producing an **accuracy of 100%.**")
                    img = Image.open("ran train.jpg")
                    st.image(img)
                
                else:
                    st.subheader("The Classification Report Testing Model of Random Forest Classifier")
                    st.write("From the **data testing**, it can be seen that the model is able to predict the data by producing an **accuracy of 78%.**")
                    img = Image.open("ran test.jpg")
                    st.image(img)
        st.write(" ")
        st.write(" ")
        # Print classification report
        st.subheader("Classification Report of Gradient Boosting Classifier")
        with st.expander("Data Train or Data Test"):
                choose = st.selectbox("View the Classification Report of Gradient Boosting Classifier Model for :", split)

                if choose == "Data Train":
                    st.subheader("The Classification Report Training Model of Gradient Boosting Classifier")
                    st.write("From the **training data**, it can be seen that the model is able to predict the data by producing an **accuracy of 82%.**")
                    img = Image.open("boost train.jpg")
                    st.image(img)
                
                else:
                    st.subheader("The Classification Report Testing Model of Gradient Boosting Classifier")
                    st.write("From the **data testing**, it can be seen that the model is able to predict the data by producing an **accuracy of 79%.**")
                    img = Image.open("boost test.jpg")
                    st.image(img)  
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Code")
        with st.expander("Click here to see the code 👉"):    
            st.code('''from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

#Build Model
#First model = logistic regression
log_model = LogisticRegression().fit(x_train,y_train)
print('\nThe Logistic Regression model formed is:', log_model)

# Predict data train
y_train_pred = log_model.predict(x_train)
# Print classification report 
print('\nClassification Report Training Model (Logistic Regression) :')
print(classification_report(y_train, y_train_pred)) #how accurate is y_train and the prediction of y_train (which is of x_train)

# Predict data test
y_test_pred = log_model.predict(x_test)
# Print classification report 
print('\nClassification Report Testing Model (Logistic Regression) :')
print(classification_report(y_test,y_test_pred))


#Second model = Random Forest Classifier
rdf_model = RandomForestClassifier().fit(x_train,y_train)
print('\nThe Random Forest Classifier model formed is:', rdf_model)

#Predict data train
y_train_pred = rdf_model.predict(x_train)
print('\nClassification Report Training Model (Random Forest Classifier) :')
print(classification_report(y_train,y_train_pred))

#Predict data test
y_test_pred = rdf_model.predict(x_test)
# Print classification report testing model
print('\nClassification Report Testing Model (Random Forest Classifier):')
print(classification_report(y_test,y_test_pred))


#Third model = Gradient Boosting Classifier
gbt_model = GradientBoostingClassifier().fit(x_train,y_train)
print('\nThe Gradient Boosting Classifier formed is:', gbt_model)

# Predict data train
y_train_pred = gbt_model.predict(x_train)
# Print classification report 
print('\nClassification Report Training Model (Gradient Boosting):')
print(classification_report(y_train,y_train_pred))

# Predict data test
y_test_pred = gbt_model.predict(x_test)
# Print classification report 
print('\nClassification Report Testing Model (Gradient Boosting):')
print(classification_report(y_test,y_test_pred))''',language='python')
    #Confusion matrix of 3 models
    elif choice == "Confusion Matrix":
        st.title("Machine Learning Model - Confusion Matrix")
        st.write("In this section, a Confusion Matrix will be presented for each model algorithm.")
        #Confusion matrix Logistic Regression
        st.subheader("Confusion Matrix of Logistic Regression")
        with st.expander("Training or Testing Model"):
            choose = st.selectbox("View the Confusion Matrix of Logistic Regression for :", model)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            #First model = logistic regression
            log_model = LogisticRegression().fit(x_train,y_train)
            # Predict data train
            y_train_pred = log_model.predict(x_train)
            # Predict data test
            y_test_pred = log_model.predict(x_test)
            pickle.dump(log_model, open('best_model_churn.pkl', 'wb'))
            if choose == "Training Model":
                # Form confusion matrix as a DataFrame for data training of LogisticRegression
                st.write('''The details are :
                \n- Guessed churn that actually did not churn were 642.
                \n- Guessed that didn't churn that didn't actually churn were 3225.
                \n- Guessed no churn that actually did churn were 648.
                \n- Guessed churn that didn't actually churn were 350.''')
                
                confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train, y_train_pred)), ('No churn', 'Churn'), ('No churn', 'Churn'))
                
                # Plot confusion matrix
                fig = plt.figure()
                heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
                heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
                heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

                plt.title('Confusion Matrix for Training Model\n(Logistic Regression\n', fontsize=18, color='darkblue')
                plt.ylabel('True label', fontsize=14)
                plt.xlabel('Predicted label', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)
            
            else:
                # Form confusion matrix as a DataFrame for data testing of LogisticRegression
                st.write('''The details are :
                \n- Guessed churn that actually did not churn were 264.
                \n- Guessed that didn't churn that didn't actually churn were 1389.
                \n- Guessed no churn that actually did churn were 282.
                \n- Guessed churn that didn't actually churn were 150.''')
                
                confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, y_test_pred)), ('No churn','Churn'), ('No churn','Churn'))

                # Plot confusion matrix
                fig = plt.figure()
                heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
                heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
                heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

                plt.title('Confusion Matrix for Testing Model\n(Logistic Regression)\n', fontsize=18, color='darkblue')
                plt.ylabel('True label', fontsize=14)
                plt.xlabel('Predicted label', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)
        st.write(" ")
        st.write(" ")

        #Confusion matrix RandomForestClassifier
        st.subheader("Confusion Matrix of Random Forest Classifier")
        with st.expander("Training or Testing Model"):
            choose = st.selectbox("View the Confusion Matrix of Random Forest Classifier for :", model)
            
            #Second model = Random Forest Classifier
            rdf_model = RandomForestClassifier().fit(x_train,y_train)
            #Predict data train
            y_train_pred = rdf_model.predict(x_train)
            #Predict data test
            y_test_pred = rdf_model.predict(x_test)
            if choose == "Training Model":
                st.write('''The details are :
                \n- Guessed churn that actually did not churn were 1274.
                \n- Guessed that didn't churn that didn't actually churn were 3570.
                \n- Guessed no churn that actually did churn were 16.
                \n- Guessed churn that didn't actually churn were 5.''')
                # Form confusion matrix as a DataFrame for data training of RandomForestClassifier
                confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train,y_train_pred)), ('No churn','Churn'), ('No churn','Churn'))
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                # Plot confusion matrix
                fig = plt.figure()
                heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
                heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
                heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

                plt.title('Confusion Matrix for Training Model\n(Random Forest)', fontsize=18, color='darkblue')
                plt.ylabel('True label', fontsize=14)
                plt.xlabel('Predicted label', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.write('''The details are :
                \n- Guessed churn that actually did not churn were 258.
                \n- Guessed that didn't churn that didn't actually churn were 1368.
                \n- Guessed no churn that actually did churn were 288.
                \n- Guessed churn that didn't actually churn were 171.''')
                # Form confusion matrix as a DataFrame for data testing of RandomForestClassifier
                confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, y_test_pred)), ('No churn','Churn'), ('No churn','Churn'))

                # Plot confusion matrix testing data
                fig = plt.figure()
                heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
                heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
                heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

                plt.title('Confusion Matrix for Testing Model\n(Random Forest)\n', fontsize=18, color='darkblue')
                plt.ylabel('True label', fontsize=14)
                plt.xlabel('Predicted label', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)
        st.write(" ")
        st.write(" ")

        #Confusion matrix Gradient Boosting Classifier
        st.subheader("Confusion Matrix of Gradient Boosting Classifier")
        with st.expander("Training or Testing Model"):
            choose = st.selectbox("View the Confusion Matrix of Gradient Boosting Classifier for :", model)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            #Third model = Gradient Boosting Classifier
            gbt_model = GradientBoostingClassifier().fit(x_train,y_train)
            # Predict data train
            y_train_pred = gbt_model.predict(x_train)
            # Predict data test
            y_test_pred = gbt_model.predict(x_test) 
            if choose == "Training Model":
                # Form confusion matrix as a DataFrame for data training of Gradient Boosting Classifier
                confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train,y_train_pred)), ('No churn','Churn'), ('No churn','Churn'))
                st.write('''The details are :
                \n- Guessed churn that actually did not churn were 684.
                \n- Guessed that didn't churn that didn't actually churn were 3286.
                \n- Guessed no churn that actually did churn were 606.
                \n- Guessed churn that didn't actually churn were 289.''')
                # Plot confusion matrix
                fig = plt.figure()
                heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
                heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
                heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

                plt.title('Confusion Matrix for Training Model\n(Gradient Boosting)', fontsize=18, color='darkblue')
                plt.ylabel('True label', fontsize=14)
                plt.xlabel('Predicted label', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)
            
            else:
                st.write('''The details are :
                \n- Guessed churn that actually did not churn were 261.
                \n- Guessed that didn't churn that didn't actually churn were 1394.
                \n- Guessed no churn that actually did churn were 285.
                \n- Guessed churn that didn't actually churn were 145.''')
                # Form confusion matrix as a DataFrame for data testing of Gradient Boosting Classifier
                confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, y_test_pred)), ('No churn','Churn'), ('No churn','Churn'))

                # Plot confusion matrix
                fig =  plt.figure()
                heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
                heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
                heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

                plt.title('Confusion Matrix for Testing Model\n(Gradient Boosting)', fontsize=18, color='darkblue')
                plt.ylabel('True label', fontsize=14)
                plt.xlabel('Predicted label', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Code")
        with st.expander("Click here to see the code 👉"):    
            st.code('''#Confusion matrix of 3 models

# Form confusion matrix as a DataFrame for data training of LogisticRegression
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train, y_train_pred)), ('No churn', 'Churn'), ('No churn', 'Churn'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Logistic Regression\n', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Form confusion matrix as a DataFrame for data testing of LogisticRegression
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, y_test_pred)), ('No churn','Churn'), ('No churn','Churn'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Logistic Regression)\n', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------

# Form confusion matrix as a DataFrame for data training of RandomForestClassifier
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train,y_train_pred)), ('No churn','Churn'), ('No churn','Churn'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Random Forest)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Form confusion matrix as a DataFrame for data testing of RandomForestClassifier
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, y_test_pred)), ('No churn','Churn'), ('No churn','Churn'))

# Plot confusion matrix testing data
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Random Forest)\n', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------
 
# Form confusion matrix as a DataFrame for data training of Gradient Boosting Classifier
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train,y_train_pred)), ('No churn','Churn'), ('No churn','Churn'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Gradient Boosting)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Form confusion matrix as a DataFrame for data testing of Gradient Boosting Classifier
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, y_test_pred)), ('No churn','Churn'), ('No churn','Churn'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Gradient Boosting)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()''',language='python')

    elif choice == "Prediction":
        st.title("Machine Learning Model - Prediction")
        st.header("Attribute List")
        st.text('''1. Gender : (Male, Female)
\n2. SeniorCitizen : (Yes, No)
\n3. Partner : (Yes, No)
\n4. Tenure : Number of months the customer has stayed with the company (0-73 Moths)
\n5. PhoneService : (Yes, No)
\n6. StreamingTV : (Yes, No)
\n7. InternetService : (Yes, No)
\n8.PaperlessBilling : (Yes, No)
\n9. MonthlyCharges : (0.000000 - 170.000000)
\n10. TotalCharges : (19.000000 - 9000.000000)''')

        st.subheader("Give the Input")

        encoded = {"Female" : 0, "Male" : 1, "No" : 0, "Yes" : 1}

        def a(val, my_dict):
            for key,value in my_dict.items():
                if val == key:
                    return value

        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("What is the customer gender?", ["Male", "Female"])
            senior = st.radio("Is a customer senior citizen or not?", ["Yes", "No"])
            partner = st.radio("Does customer have partners?", ["Yes","No"])
            tenure = st.number_input("Tenure : Number of months the customer has stayed with the company", 0, 73, 1)
            

        with col2:
            phone = st.radio("Does customer have phone service or not?", ["Yes", "No"])
            streaming = st.radio("Does customer have streaming TV or not?", ["Yes",'No'])
            internet = st.radio("Does customer have internet service provider?", ["Yes","No"])
            paperless = st.radio("Does customer have paperless billing?", ["Yes",'No'])
        
        monthly = st.number_input("Monthly Charge : The amount charged to the customer monthly", 0.000000, 170.000000)
        total = st.number_input("Total Charge : The total amount charged to the customer", 0, 9000.000000)
        st.write(" ")
        st.write(" ")
        st.write(" ")
        with st.expander("The selected option"):
            so = {"Gender" : gender, "Senior Citizen" : senior, "Partner" : partner, "Tenure" : tenure, "Phone Service" : phone
            , "Streaming TV" : streaming, "Internet Srvice" : internet, "Paperless Billing" : paperless, "Monthly Charge" : monthly
            , "Total Charge" : total}
            st.write(so)
        
            result = []

            for i in so.values():
                if type(i) == int or type(i) == float:
                    result.append(i)
                else:
                    res = a(i, encoded)
                    result.append(res)
            
            st.write(result)

            input = np.array(result).reshape(1,-1)
            st.write(input)
        
        with st.expander("Prediction Result"):

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            #First model = logistic regression
            log_model = LogisticRegression().fit(x_train,y_train)
            # Predict data train
            y_train_pred = log_model.predict(x_train)
            # Predict data test
            y_test_pred = log_model.predict(x_test)

            file = open('best_model_churn', 'wb')
            joblib.dump(log_model, file)
            file.close()
            
            data = joblib.load('best_model_churn')
            prediction = data.predict(input)

            prob = data.predict_proba(input)

            if prediction == 0:
                st.success("Customer has a Tendency to **not Churn**")
                prob_score = {"Probability to not Churn" : prob[0][0], "Probability to Churn" : prob[0][1]}
                st.write(prob_score)

            else:
                st.warning("Customer has a Tendency to **Churn**")
                prob_score = {"Probability to Churn" : prob[0][1], "Probability to not Churn" : prob[0][0]}
                st.write(prob_score)
    else:
        st.title("Machine Learning Model - Conclusion")
        st.header("Conclusion")
        st.write('''Based on the modeling that has been done using Logistic Regression, Random Forest and Gradiant Boost, it can be concluded that to predict churn from telco customers using this dataset the best model is the **Logistic Regression algorithm**.

\nThis is because the performance of the Logistic Regression model tends to be able to predict equally well in the training and testing phases **(80% training accuracy, 79% testing accuracy)**, on the other hand, other algorithms tend to over-fitting their performance.

\nHowever, this does not make us draw the conclusion that if to do any modeling we use Logistic Regression, we still have to do a lot of model experiments to determine which one is the best.''')

        