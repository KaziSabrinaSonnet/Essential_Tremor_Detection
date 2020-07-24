
#%%
import pandas as pd
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
#%%
"""
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
"""
#%%
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


#%%
def velocity_extraction(df):
    Velocity = []
    horizontal_velocity = []
    horizontal_velocity_magnitude = []
    vertical_velocity = []
    vertical_velocity_magnitude = []
    velocity_magnitude = []
    TimeStamp_difference =  []
    timestamps = []
    lookahead = 3
    t = 0
    for i in range(len(df)-lookahead):
        if t+lookahead <= len(df)-1:
            Velocity.append(((df['XEnd'][t+lookahead] - df['XStart'][t])/(df['TimeStamp'][t+lookahead]-df['TimeStamp'][t]) , (df['YEnd'][t+lookahead]-df['YStart'][t])/(df['TimeStamp'][t+lookahead]-df['TimeStamp'][t])))
            horizontal_velocity.append((df['XEnd'][t+lookahead] - df['XStart'][t])/(df['TimeStamp'][t+lookahead]-df['TimeStamp'][t]))
            vertical_velocity.append((df['YEnd'][t+lookahead] - df['YStart'][t])/(df['TimeStamp'][t+lookahead]-df['TimeStamp'][t]))
            velocity_magnitude.append(math.sqrt(((df['XEnd'][t+lookahead]-df['XStart'][t])/(df['TimeStamp'][t+lookahead]-df['TimeStamp'][t]))**2 + (((df['YEnd'][t+lookahead]-df['YStart'][t])/(df['TimeStamp'][t+lookahead]-df['TimeStamp'][t]))**2)))
            TimeStamp_difference.append(df['TimeStamp'][t+lookahead]-df['TimeStamp'][t])
            horizontal_velocity_magnitude.append(abs(horizontal_velocity[len(horizontal_velocity)-1]))
            vertical_velocity_magnitude.append(abs(vertical_velocity[len(vertical_velocity)-1]))
            t = t+lookahead
            timestamps.append(df['TimeStamp'][t])
        else:
            break
    mean_velocity_magnitude = np.mean(velocity_magnitude)  
    mean_horizontal_velocity_magnitude= np.mean(horizontal_velocity_magnitude)
    mean_vertical_velocity_magnitude= np.mean(vertical_velocity_magnitude)
    return Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps

def acceleration_extraction(df):
    Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
    acceleration = []
    horizontal_acceleration =  []
    vertical_acceleration = []
    acceleration_magnitude = []
    horizontal_acceleration_magnitude = []
    vertical_acceleration_magnitude = []
    timestamps = []
    t= 0
    lookahead=1
    for i in range(len(Velocity)-lookahead):
        if t+lookahead <= len(Velocity)-1:
            acceleration.append(((Velocity[t+lookahead][0]-Velocity[t][0])/TimeStamp_difference[t] , (Velocity[t+lookahead][1]-Velocity[t][1])/TimeStamp_difference[t]))
            horizontal_acceleration.append((horizontal_velocity[t+lookahead]-horizontal_velocity[t])/TimeStamp_difference[t])
            vertical_acceleration.append((vertical_velocity[t+lookahead]-vertical_velocity[t])/TimeStamp_difference[t])
            horizontal_acceleration_magnitude.append(abs(horizontal_acceleration[len(horizontal_acceleration)-1]))
            vertical_acceleration_magnitude.append(abs(vertical_acceleration[len(vertical_acceleration)-1]))
            acceleration_magnitude.append(math.sqrt(((Velocity[t+lookahead][0]-Velocity[t][0])/TimeStamp_difference[t])**2 + ((Velocity[t+lookahead][1]-Velocity[t][1])/TimeStamp_difference[t])**2))
            t= t+lookahead
            timestamps.append(df['TimeStamp'][t])
        else:
            break
      

    mean_acceleration_magnitude = np.mean(acceleration_magnitude)  
    mean_horizontal_acceleration_magnitude = np.mean(horizontal_acceleration_magnitude)
    mean_vertical_acceleration_magnitude = np.mean(vertical_acceleration_magnitude)
    return acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps
def NCV_per_halfcircle(df):
    Vel = []
    ncv = []
    temp_ncv = 0
    basex = df['XStart'][0]
    for i in range(len(df)-2):
        if df['XStart'][i] == basex:
            ncv.append(temp_ncv)
            temp_ncv = 0
            continue
            
        Vel.append(((df['XEnd'][i+1] - df['XStart'][i])/(df['TimeStamp'][i+1]-df['TimeStamp'][i]) , (df['YEnd'][i+1]-df['YStart'][i])/(df['TimeStamp'][i+1]-df['TimeStamp'][i])))
        if Vel[len(Vel)-1] != (0,0):
            temp_ncv+=1
    ncv.append(temp_ncv)
    ncv = list(filter((2).__ne__, ncv))
    ncv_Val = np.sum(ncv)/np.count_nonzero(ncv)
    return ncv,ncv_Val
def NCA_per_halfcircle(df):
    accl = []
    nca = []
    temp_nca = 0
    basex = df['XStart'][0]
    for i in range(len(Velocity)-2):
        if df['XStart'][i] == basex:
            nca.append(temp_nca)
            temp_nca = 0
            continue
            
        accl.append(((Velocity[i+1][0]-Velocity[i][0])/TimeStamp_difference[i] , (Velocity[i+1][1]-Velocity[i][1])/TimeStamp_difference[i]))
        if accl[len(accl)-1] != (0,0):
            temp_nca+=1
    nca.append(temp_nca)
    nca = list(filter((2).__ne__, nca))
    nca_Val = np.sum(nca)/np.count_nonzero(nca)
    return nca,nca_Val
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
NCA_Tremor= []
NCA_VAL_Tremor= []
NCV_Tremor= []
NCV_VAL_Tremor= []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        ncv,ncv_Val = NCV_per_halfcircle(df)
        NCV_Tremor.append(ncv)
        NCV_VAL_Tremor.append(ncv_Val)
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
NCA_Healthy= []
NCA_VAL_Healthy= []
NCV_Healthy= []
NCV_VAL_Healthy= []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        ncv,ncv_Val = NCV_per_halfcircle(df)
        NCV_Healthy.append(ncv)
        NCV_VAL_Healthy.append(ncv_Val)
#%%
import matplotlib.pyplot as plt
"""
plt.plot(NCV_VAL_Healthy,'g*', NCV_VAL_Tremor, 'ro')
plt.xticks([])
plt.title('Number of Time Velocity Changes (Spirals)')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(NCV_VAL_Healthy)), NCV_VAL_Healthy, 'g*')
plt.plot(np.zeros(np.shape(NCV_VAL_Tremor)), NCV_VAL_Tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Number of Time Velocity Changes (Lines)')
plt.show()

#%%
