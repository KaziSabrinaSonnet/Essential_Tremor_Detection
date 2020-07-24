#%%
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
dataset = pd.read_csv('SpiralLineAnalysis.csv')
X = dataset.iloc[:, 0:29].values
y = dataset.loc[:,['Label']].values
#%%
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
lda_object = lda.fit(X_train, y_train)

#%%
#Applyting LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
#Fitting logistic regression to the training set 
#%%
#Logistic regression linear classifier
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train, y_train) 
#%%
#predict the test set result 
y_pred= classifier.predict(X_test)

#%%
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
#%%
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_train[:,0],
    X_train[:,1],
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)

#%%
