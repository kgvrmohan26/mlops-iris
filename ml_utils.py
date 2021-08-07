#rom sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import subprocess


# define the class encodings and reverse encodings
classes = {0: "Bad", 1: "Good"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    #Load data set from https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
    df=read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",sep=" ",header=None)
    y=df[20]
    X=df.drop(20,axis=1)
    # select categorical and numerical features
    cat_ix=X.select_dtypes(include=['object','bool']).columns
    ct=ColumnTransformer([('o',OneHotEncoder(),cat_ix)],remainder='passthrough')
    clf = GaussianNB()
    global pipe
    pipe=Pipeline([("ct",ct),("clf",clf)])
    
    #Label encode the target variable to have  the classes 0 and 1 
    y=LabelEncoder().fit_transform(y)
    
    #Do the Split and train the model
    X_train, X_test, y_train, y_test= train_test_split(X ,y,test_size=0.2)
    pipe.fit( X_train,y_train)
    
    #calculate and print accuracy_score
    acc=accuracy_score(y_test,pipe.predict(X_test))
    print(f"Model trained with accuracy:{round(acc,3)}")
    
    #Generating Explainability File
    #subprocess.call(["jupyter","nbconvert","--to","notebook","--execute","explainable_AI_starter.ipynb"])
    #subprocess.call(["jupyter","nbconvert","explainable_AI_starter.ipynb","--no-input","--to","html"])
    print ("Explainability File Generated")
    
    


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    X=pd.DataFrame([x])
    global pipe
    prediction = pipe.predict(X)[0]
    print(f"Model prediction: {classes[prediction]}")
    return (classes[prediction])

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.loan] for d in data]

    # fit the classifier again based on the new data obtained
    global pipe
    pipe.fit(X,y)
    


