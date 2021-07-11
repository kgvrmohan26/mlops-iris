from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# define a Gaussain NB classifier
clf = GaussianNB()

#define a Linear classifier
clf_Lreg = LogisticRegression(penalty='l2',C=1.0, max_iter=10000)


# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    
    global clf
    global clf_Lreg
    
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train) # Fit Gaussain classifier
    
    clf_Lreg.fit(X_train, y_train) # Fit Linear classifier

    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf.predict(X_test))  # Gaussain classifier accuracy
    acc_Lreg = accuracy_score(y_test, clf_Lreg.predict(X_test))  # Linear classifier accuracy
    print(f"Model GaussianNB classifier trained with accuracy: {round(acc, 3)}")
    print(f"Model LogisticRegression classifier trained with accuracy: {round(acc_Lreg, 3)}")
    if (acc < acc_Lreg): #if LogisticRegression accuracy better than GaussianNB
       clf=clf_Lreg #Assign LogisticRegression classifier to clf
       print ('LogisticRegression got better accuracy')
       print('classifier used for prediction is',clf)
    else: #if GaussianNB is better/equal than/to LogisticRegression
       print ('GaussianNB got better/equal accuracy')
       print('classifier used for prediction is',clf)#clf contain GaussianNB classifier as assigned initially
    


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return (classes[prediction])

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
