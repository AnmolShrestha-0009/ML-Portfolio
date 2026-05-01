from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib 
from sklearn.linear_model import LogisticRegression
import pandas as pd
df=load_iris(as_frame=True).frame
col=[]
cols=df.columns
user_input=[]
##user_input
for label in cols[:-1]:
    x=int(input(f"Enter the {label}"))
    user_input.append(x)
user_input=np.array(user_input).reshape(1,-1)
X, y=load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe=Pipeline([
    ('scalar',StandardScaler()),
    ('model',LogisticRegression())
])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
score=cross_val_score(pipe,X,y,cv=5)
print(f"The mean accuracy of  the model is {score.mean()}")  
print(f"The classfication report is {classification_report(y_test,y_pred)}")
#predictinfg as per user requirement
pre=pipe.predict(user_input)
print(f"The probable output is{pre}")
joblib.dump(pipe,"iris.pkl")

