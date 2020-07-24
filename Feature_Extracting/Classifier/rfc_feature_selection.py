#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
#%%
# Importing the dataset
dataset = pd.read_csv('Spiral_Line_Analysis.csv')
X = dataset.iloc[:, 1:30].values
y = dataset.loc[:,['Label']].values
#%%
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
#%%
y_train = y_train.ravel()
y_test = y_test.ravel()

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)

# %%
# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
sfs1 = sfs(clf, k_features=5, forward=True, floating=False, verbose=2, scoring='accuracy', cv=5)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)

# %%
# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

# %%
# Build full model on ALL features, for comparison
clf = LogisticRegressionClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))

# %%
