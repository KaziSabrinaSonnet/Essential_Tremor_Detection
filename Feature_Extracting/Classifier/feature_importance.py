#%%
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importing the dataset
dataset = pd.read_csv('SpiralLineAnalysis.csv')
X = dataset.iloc[:, 0:30].values
y = dataset.loc[:,['Label']].values
#%%
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#%%
from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression(fit_intercept=False)
logreg.fit(X_train, y_train) 
#%%
np.round(logreg.coef_, decimals=2)

# %%
