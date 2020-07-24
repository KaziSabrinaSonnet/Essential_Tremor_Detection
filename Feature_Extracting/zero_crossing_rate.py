#%%
import pandas as pd
import numpy as np
import os
import seaborn as sns
from numpy import int64
from matplotlib import pyplot as plt
import math
from scipy import signal
import matplotlib.pyplot as plt
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
#%%
def read_data_file(path_file):
    df = pd.DataFrame(columns=COLUMN_NAMES)
    with open(path_file, 'r') as fp:
        for line in fp:
            line_splits = line.split(',')
            # Ignore first line
            if len(line_splits) == 5:
                data_row = []
                for token in line_splits[0].split('-'):
                    data_row.append(int(token))  
                for token in line_splits[1:]:
                    data_row.append(int(token))          
                df = df.append(pd.Series(data_row, index=df.columns ), ignore_index=True)
                reference_timestamp = (df['TimeStamp'][0])* (10**(-6))
    df['TimeStamp'] = (df['TimeStamp'])* (10**(-6)) - reference_timestamp
    return df

def create_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
#%%
def calculate_theta(x, y):
    theta = (math.atan2(y, x))
    return theta

def caculate_radius(x, y):
    radius = math.sqrt(((x)**2+(y)**2))
    return radius
#%%
def unravel_spiral(df):
    radius = []
    theta = [] 
    t=0
    lookahead = 50
    n= len(df)//lookahead
    s= (n-1)*lookahead
    p= (n*lookahead)
    df.iloc[s:p]['XEnd'] = 0
    df.iloc[s:p]['YEnd'] = 0
    for i in range(len(df)-lookahead):
        if t+lookahead <= len(df)-1:
            radius.append(caculate_radius(df['XEnd'][t], df['YEnd'][t]))
            theta.append(calculate_theta(df['XEnd'][t], df['YEnd'][t]))
            t= t+lookahead
        else: 
            break
    return radius, theta

def plot_unravelled_spiral(df, path_file):
        fig = sns.lineplot(df['Angle'], df['Radius'])
        fig.set(xlabel='Angle', ylabel='Radius')
        path_folder = os.path.join(data_path, "plots_unravelled_spirals")
        create_folder(path_folder)
        fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_unravelling.png'))
        plt.title("Unravelled Spiral(Healthy)")
        plt.show()
#%%
import numpy as np
from random import sample 
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        df_polar_coordinate['Radius'] = radius
        Tremor_unravel= df_polar_coordinate['Radius'].tolist()
        zero_x = 0
        for idx, item in enumerate(Tremor_unravel[:-1]):
            if Tremor_unravel[idx] < 0 and Tremor_unravel[idx+1] > 0:
                zero_x +=1
            if Tremor_unravel[idx] > 0 and Tremor_unravel[idx+1] < 0:
                zero_x +=1
#%%

