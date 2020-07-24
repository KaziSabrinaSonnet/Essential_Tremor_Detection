#%%
import pandas as pd 
# Importing the dataset
dataset = pd.read_csv('Spiral_Line2.csv')
X = dataset.iloc[:, 0:30].values
y = dataset.loc[:,['Label']].values

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

#%%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(logreg, X, y, cv=10)
A= accuracy_score(y, predicted) 

# %%
