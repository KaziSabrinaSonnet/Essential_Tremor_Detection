#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
#%%
# Importing the dataset
dataset = pd.read_csv('Features1.csv')
X = dataset.iloc[:, 1:18].values
Y = dataset.loc[:,['Label']].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 8)
#%%
def generate_accuracy_and_heatmap(model, x, y):
#     cm = confusion_matrix(y,model.predict(x))
#     sns.heatmap(cm,annot=True,fmt="d")
    ac = accuracy_score(y,model.predict(x))
    print('Accuracy is: ', ac)
    print ("\n")
    return 1
#%%
clf_lr = LogisticRegression()      
lr_baseline_model = clf_lr.fit(x_train,y_train)
generate_accuracy_and_heatmap(lr_baseline_model, x_test, y_test)

#%%
select_feature = SelectKBest(chi2, k=3).fit(x_train, y_train)
x_train_chi = select_feature.transform(x_train)
x_test_chi = select_feature.transform(x_test)
# %%
lr_chi_model = clf_lr.fit(x_train_chi,y_train)

# %%
rfe = RFE(estimator=clf_lr, step=1)
rfe = rfe.fit(x_train, y_train)
x_train_rfe = rfe.transform(x_train)
x_test_rfe = rfe.transform(x_test)
lr_rfe_model = clf_lr.fit(x_train_rfe, y_train)

# %%
rfecv = RFECV(estimator=clf_lr, step=1, cv=5, scoring='accuracy')
rfecv = rfecv.fit(x_train, y_train)

#%%
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validated Accuracy")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_,'-o')
plt.title("Recursive Feature Elimination with Cross Validation")
plt.show()
plt.savefig("D:\\Autumn 2019\\Research\\Feature_Extracting\\StimOn")

# %%
