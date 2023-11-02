import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression    #Test this model if you wish
#from sklearn.tree import DecisionTreeClassifier         #Test this model if you wish
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import datasets                           #Can also import Iris from sklearn datasets

#Create dataframe object from dataset using Pandas
df = pd.read_csv('Iris.csv')    #Extract Iris.csv file to current directory, file taken from "https://www.kaggle.com/datasets/uciml/iris/"

#print(type(df)) #uncomment to check what data type df is

def show_dataset_info_using_pandas(dataframe):   
    print(dataframe.head())
    print("--------------------------------------------------------------------------")
    print(dataframe.describe())
    print("--------------------------------------------------------------------------")
    print(dataframe.info())
    print("--------------------------------------------------------------------------")
    print(dataframe['Species'].value_counts())

#show_dataset_info_using_pandas(df)       #uncomment to check dataset info


#Label Encoding, converting output label into numeric form, such as replacing "Iris-setosa" to 0, "Iris-veriscolor" to 1 etc.
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
#print(df.head())   #uncomment to check output label converted to numeric form

#Dataset split for training and testing, 50:50 split training_data:test_data

X = df.drop(columns=['Species', 'Id'])    # Remove 'Specices' and 'Id' column when training, X refers to features here
Y = df['Species']                         # Y contains output label only, Y refers to the label here

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)       #50% used for test, 50% used for training

#Confirming training and test data are as expected
"""
print(x_train.shape)        #first value number of rows of training data, second value number of features
print(y_train.shape)        
print(x_test.shape)         #first value number of rows of test data, second value number of features         
print(y_test.shape)
"""

#Using this model for classification
model = KNeighborsClassifier()   

#Train the model using training data
model.fit(x_train, y_train)

#print(x_test.head())
print(f"Accuracy of our Model is: {model.score(x_test, y_test)*100}%")


#Manually create test data and ask for prediction, array values are: [SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]
test_feature_values = np.array([5.8,4.0,1.2,0.2])  # Example feature values, change these to what you wish to test the model

# Use the trained model to predict the class label
predicted_numeric_output_form = model.predict([test_feature_values])

# Map the numeric class label back to the original class names
predicted_word_output_form = le.inverse_transform(predicted_numeric_output_form)

#print(predicted_word_output_form
print("Prediction:", predicted_word_output_form[0])

