#%%
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
#Importing the dataset
dataset = pd.read_csv('Spiral_Line.csv')
#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(25,25))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()
 
    
correlation_heatmap(dataset)

# %%
