limport pandas as pd
import numpy as np
import random
import os
import seaborn as sns
from numpy import int64
from matplotlib import pyplot as plt
import math
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy import fft, arange
from random import random
#%% 
# Read spirals
def read_data_file(path_file):
    df = pd.DataFrame(columns=COLUMN_NAMES)
    with open(path_file, 'r') as fp:
        for line in fp:
            if 'Lines Begun' in line:
                break
            line_splits = line.split(',')
            # Ignore first line
            if len(line_splits) == 5:
                data_row = []
                for token in line_splits[0].split('-'):
                    data_row.append(int(token))  
                for token in line_splits[1:]:
                    data_row.append(int(token))          
                df = df.append(pd.Series(data_row, index=df.columns ), ignore_index=True)
    print("File: " + path_file)
    print("Timestamp Length: " + str(len(df['TimeStamp'].tolist())))
    df['TimeStamp'] = (df['TimeStamp'])* (10**(-6)) - (df['TimeStamp'][0])* (10**(-6))
    return df

def create_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

#%%
"""
# Read for line
def read_data_file(path_file):
    df = pd.DataFrame(columns=COLUMN_NAMES)
    bool_extract_data = False
    with open(path_file, 'r') as fp:
        for line in fp:
            if not bool_extract_data and 'Lines Begun' in line:
                bool_extract_data = True
                continue
            if bool_extract_data:
                line_splits = line.split(',')
                # Ignore first line
                if len(line_splits) == 5:
                    data_row = []
                    for token in line_splits[0].split('-'):
                        data_row.append(int(token))  
                    for token in line_splits[1:]:
                        data_row.append(int(token))          
                    df = df.append(pd.Series(data_row, index=df.columns ), ignore_index=True)
    print("File: " + path_file)
    print("Timestamp Length: " + str(len(df['TimeStamp'].tolist())))
    if not bool_extract_data:
        print("No Lines data in file: " + path_file)
    df['TimeStamp'] = (df['TimeStamp'])* (10**(-6)) - (df['TimeStamp'][0])* (10**(-6))
    return df

def create_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

"""

#%%
def plot_pressure(df, path_file):
    fig = sns.tsplot(df['Pressure'], df['TimeStamp'])
    fig.set(xlabel='Time', ylabel='Pressure')
    path_folder = os.path.join(data_path, "plots_pressure")
    create_folder(path_folder)
    fig.get_figure().savefig(
        os.path.join(path_folder,
        os.path.basename(path_file) + '_pressure.png'))
    plt.show()

#%%
base_folder = "D:\\Masters@UW\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
Healthy_Pressure = []
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        df_pressure_healthy = df['Pressure'].tolist()
        Healthy_Pressure.append(np.mean(df_pressure_healthy))
        plot_pressure(df, file)
      
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
Tremor_Pressure = [] 
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        df_pressure_tremor = df['Pressure'].tolist()
        Tremor_Pressure.append(np.mean(df_pressure_tremor))
        plot_pressure(df, file)
        
#%%
import matplotlib.pyplot as plt
"""
plt.plot(Frac_Healthy,'g*', Frac_Tremor, 'ro')
plt.xticks([])
plt.title('Fractral Dimension')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(Healthy_Pressure)), Healthy_Pressure, 'g*')
plt.plot(np.zeros(np.shape(Tremor_Pressure)), Tremor_Pressure, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Pressure (Lines)')
plt.show()
#%%
base_folder = "D:\\Masters@UW\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'Pressure1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        df_pressure_untreated = df['Pressure'].tolist()
#%%
base_folder = "D:\\Masters@UW\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'Pressure2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        df_pressure_healthy = df['Pressure'].tolist()
#%%
base_folder = "D:\\Masters@UW\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'Pressure3')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        df_pressure_treated = df['Pressure'].tolist()
#%%
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(len(df_pressure_healthy))
y= np.arange(len(df_pressure_untreated[0:1600]))
z= np.arange(len(df_pressure_treated[0:1600]))
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(x,df_pressure_healthy,c='g',label='Healthy')
ax.plot(z,df_pressure_treated[0:1600],c='b', label='Treated')
ax.plot(y,df_pressure_untreated[0:1600],c='r', label='Untreated')
plt.xlabel("Sample Points")
plt.ylabel("Pressure")
plt.legend(loc=4, prop={'size':10})
plt.title("Pressure Analysis")
plt.draw()
# %%
