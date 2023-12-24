#importing necessary libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
import pickle

#importing data
df = pd.read_csv("data.csv")

#dropping these attributes as we would be using only numerical values
df.drop(["Id"], axis = 1, inplace = True)
df.drop(["Profession"], axis=1, inplace = True)
df.drop(["CITY"], axis=1, inplace = True)
df.drop(["STATE"], axis=1, inplace = True)

#replacing these attributes by numerical values so that we could use these attributes in our model
df.replace({'Married/Single': {'married': 1, 'single': 0}},inplace=True)
df.replace({'House_Ownership': {'owned': 1, 'rented': 0, "norent_noown":-1}},inplace=True)
df.replace({'Car_Ownership': {'yes': 1, 'no': 0}},inplace=True)

#train test splitting
X = df.drop(['Risk_Flag'],axis=1)
Y = df.Risk_Flag
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Looking for correlations
corr_matrix = df.corr()
data=dict(corr_matrix["Risk_Flag"].sort_values(ascending = True))
data.pop("Risk_Flag")

#Since, "Car_Ownership", "Experience", "Married/Single" and "Age" showed highly negative correlation with respect to "Risk_Flag", we decided to use these features in our regression model.
imp_features = ["Car_Ownership","Experience","Married/Single","Age"]

df_for_pca = df[imp_features]
label_encoder = LabelEncoder()

for col in imp_features:
    df[col] = label_encoder.fit_transform(df[col])

#Scorecard - In order to provide an equal opportunity for everyone to establish credit, we prepared a scorecard which evaluates the features of the dataframe and produces a score which can be used to assess their credibility.
df["Scorecard"]=0
for i in data:
    df["Scorecard"]+=abs(data[i])*df[i]

#After checking the correlation of "Scorecard" with "Risk_Flag", we came to the conclusion that having a high scorecard value means reduced risk for the loan rejection.
#So, the people having no prior credit score can assess their credibility via this score.

#Applying the model
sm = SMOTE(random_state = 42)
X_resample, Y_resample = sm.fit_resample(X_train, Y_train)

#As there is a lot of imbalance between the amount of people who are eligible for availing loan and those who are not, therefore we decided to use SMOTE (Synthetic Minority Oversampling Technique) to eradicate this disparity.
#After removing all the redundancies, we applied our model on test data to test its accuracy.
model_used = LogisticRegression()
model_used.fit(X_resample, Y_resample)
Y_pred = model_used.predict(X_test)
model_accuracy = (model_used.score(X_test, Y_test))*100

#The accuracy of the model is: 87.59325396825398 %

pickle.dump(model_used, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

