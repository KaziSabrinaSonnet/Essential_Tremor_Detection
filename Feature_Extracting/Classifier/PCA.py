#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importing the dataset
dataset = pd.read_csv('Spiral_Line_Analysis.csv')
X = dataset.iloc[:, 1:30].values
y = dataset.loc[:,['Label']].values
#%%
from sklearn.preprocessing import StandardScaler
# Standardizing the features
X = StandardScaler().fit_transform(X)

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#%%
finalDf = pd.concat([principalDf, dataset[['Label']]] , axis = 1)

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = principalDf.values
y_list = [y[i][0] for i in range(y.shape[0])]
class_labels = set(y_list)

lda = LinearDiscriminantAnalysis()
lda_object = lda.fit(X, y)

#%%
from matplotlib.lines import Line2D 
from matplotlib.patches import Circle
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA and LDA', fontsize = 20)
targets = ['Tremor', 'Healthy']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

x1 = np.array([np.min(X[:,0], axis=0), np.max(X[:,0], axis=0)])

b = lda.intercept_[0]
m = lda.coef_[0][0]
n = lda.coef_[0][1]
y1 = -(b + m * x1)/n
ax.plot(x1, y1)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Healthy',
                          markerfacecolor='g', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Tremor',
                          markerfacecolor='r', markersize=10)]

ax.legend(handles=legend_elements)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.grid()

# %%
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = principalDf.values
y_list = [y[i][0] for i in range(y.shape[0])]
class_labels = set(y_list)

lda = LinearDiscriminantAnalysis()
lda_object = lda.fit(X, y)

plt.xlim(np.min(X[:, 0])-2.0, np.max(X[:, 0]) + 2.0)
plt.xlim(np.min(X[:, 1])-2.0, np.max(X[:, 1]) + 2.0)
# Plot the hyperplanes
for l,c,m in zip(np.unique(y),['green','red'],['s','x']):
    #indices = [i for i in range(len(y_list)) if y_list[i] == l]
    plt.scatter(X[y_list==l, 0],
                X[y_list==l, 1],
                c=c, marker=m, label=l, s=50)
plt.show()
x1 = np.array([np.min(X[:,0], axis=0), np.max(X[:,0], axis=0)])

b = lda.intercept_[0]
m = lda.coef_[0]
y1 = -(b + m * x1)
plt.plot(x1, y1)
"""
# %%
