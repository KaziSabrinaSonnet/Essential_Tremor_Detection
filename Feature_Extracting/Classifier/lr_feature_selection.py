#%%
import pandas as pd
import numpy as np
#%%
dataframe= pd.read_csv('Spiral_Line_Analysis.csv')
dataframe.head()
#%%
# Importing the dataset
dataset = pd.read_csv('Spiral_Line_Analysis.csv')
X = dataset.iloc[:, 1:30].values
y = dataset.loc[:,['Label']].values
#%%
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# %%
# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])

# %%
# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# %%
# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)

print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
#%%