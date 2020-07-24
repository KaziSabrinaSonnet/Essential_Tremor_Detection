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
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]

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
    for i in range(len(df)-2):
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

        
#%%
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
    return acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps, TimeStamp_difference
#%%
def find_jerk(df):
    acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps, TimeStamp_difference= acceleration_extraction(df)
    jerk = []
    hrz_jerk = []
    vert_jerk = []
    magnitude = []
    horz_jerk_mag = []
    vert_jerk_mag = []
    timestamps = []
    for i in range(len(acceleration)-2):
        jerk.append(((acceleration[i+1][0]-acceleration[i][0])/TimeStamp_difference[i] , (acceleration[i+1][1]-acceleration[i][1])/TimeStamp_difference[i]))
        hrz_jerk.append((horizontal_acceleration[i+1]-horizontal_acceleration[i])/TimeStamp_difference[i])
        vert_jerk.append((vertical_acceleration[i+1]-vertical_acceleration[i])/TimeStamp_difference[i])
        horz_jerk_mag.append(abs(hrz_jerk[len(hrz_jerk)-1]))
        vert_jerk_mag.append(abs(vert_jerk[len(vert_jerk)-1]))
        magnitude.append(math.sqrt(((acceleration[i+1][0]-acceleration[i][0])/TimeStamp_difference[i])**2 + ((acceleration[i+1][1]-acceleration[i][1])/TimeStamp_difference[i])**2))
        timestamps.append(df['TimeStamp'][i])
        
    magnitude_jerk = np.mean(magnitude)  
    magnitude_horz_jerk = np.mean(horz_jerk_mag)
    magnitude_vert_jerk = np.mean(vert_jerk_mag)
    print (magnitude_jerk , ' ' ,magnitude_horz_jerk, ' ',magnitude_vert_jerk )
    return jerk,magnitude,hrz_jerk,vert_jerk,TimeStamp_difference,magnitude_jerk,magnitude_horz_jerk,magnitude_vert_jerk, timestamps

#%%
from random import sample 
import hfda
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting" 
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Tremor_Jerk = []
Frac_Tremor =[]
def plot_Jerk(df, path_file):
    fig = sns.tsplot(df['Jerk'],df['TimeStamp'])
    fig.set(xlabel='Time', ylabel='Jerk', ylim= (0, 1000000))
    path_folder = os.path.join(data_path, "plots_Jerk")
    create_folder(path_folder)
    fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_Jerk.png'))
    plt.show()
    
  
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        trem_jerk,trem_magnitude,trem_hrz_jerk,trem_vert_jerk, trem_TimeStamp_difference , trem_magnitude_jerk, trem_magnitude_horz_jerk, trem_magnitude_vert_jerk, trem_timestamps =find_jerk(df) 
        df_jerk = pd.DataFrame(columns=['TimeStamp', 'Jerk'])
        df_jerk['TimeStamp'] = trem_timestamps
        df_jerk['Jerk'] = trem_magnitude
        #plot_Jerk(df_jerk, file)
        D = hfda.measure(np.array(trem_magnitude), 4)
        Frac_Tremor.append(D)
        Tremor_Jerk.append(np.mean(trem_magnitude))
#%%
from random import sample 
import hfda
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting" 
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Healthy_Jerk= []
Frac_Healthy = []
def plot_Jerk(df, path_file):
    fig = sns.tsplot(df['Jerk'], df['TimeStamp'])
    fig.set(xlabel='Time', ylabel='Jerk', ylim= (0, 1000000))
    path_folder = os.path.join(data_path, "plots_Jerk")
    create_folder(path_folder)
    fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_Jerk.png'))
    plt.show()
    
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        heal_jerk,heal_magnitude,heal_hrz_jerk,heal_vert_jerk, heal_TimeStamp_difference , heal_magnitude_jerk, heal_magnitude_horz_jerk, heal_magnitude_vert_jerk, heal_timestamps =find_jerk(df) 
        df_jerk = pd.DataFrame(columns=['TimeStamp', 'Jerk'])
        df_jerk['TimeStamp'] = heal_timestamps
        df_jerk['Jerk'] = heal_magnitude
        #plot_Jerk(df_jerk, file)
        D = hfda.measure(np.array(heal_magnitude), 4)
        Frac_Healthy.append(D)
        Healthy_Jerk.append(np.mean(heal_magnitude))
#%%
import matplotlib.pyplot as plt
"""
plt.plot(Healthy_Jerk,'g*', Tremor_Jerk, 'ro')
plt.xticks([])
plt.title('Jerk')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(Healthy_Jerk)), Healthy_Jerk, 'g*')
plt.plot(np.zeros(np.shape(Tremor_Jerk)), Tremor_Jerk, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Jerk (Spirals)')
plt.show()
#%%
def calculate_sampling_frequency(df):
    jerk,magnitude,hrz_jerk,vert_jerk,TimeStamp_difference,magnitude_jerk,magnitude_horz_jerk,magnitude_vert_jerk, timestamps = find_jerk(df)
    df_jerk = pd.DataFrame(columns=['TimeStamp', 'Jerk'])
    df_jerk['Jerk'] = magnitude
    df_list1= df_jerk['Jerk'].tolist()
    number_of_samples= len(df_list1)
    df_list2= df['TimeStamp'].tolist()
    total_time= (df_list2[len(df_list2)-1]-df_list2[0]) 
    sampling_frequency= number_of_samples/total_time
    return sampling_frequency
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_jerk_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData1\\Spectrum_Jerk'
create_folder(path_jerk_spectrum_plots)
tremor_amplitude = []
peak_list_tremor=[]
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        trem_jerk,trem_magnitude,trem_hrz_jerk,trem_vert_jerk, trem_TimeStamp_difference , trem_magnitude_jerk, trem_magnitude_horz_jerk, trem_magnitude_vert_jerk, trem_timestamps =find_jerk(df) 
        df_jerk = pd.DataFrame(columns=['TimeStamp', 'Jerk'])
        df_jerk['Jerk'] = trem_magnitude
        data=df_jerk['Jerk'].tolist()
        Fs= 133
        win = 4* Fs
        freqs, psd = signal.welch(data, Fs, nperseg=win)
        peaks, _ = find_peaks(psd, height=np.amax(psd)*0.05)
        peak_list_tremor.append(peaks)
        plt.plot(psd)
        plt.plot(peaks, psd[peaks], "x")
        plt.plot(np.zeros_like(psd), "--", color="gray")
        plt.savefig(os.path.join(path_jerk_spectrum_plots,os.path.basename(file) + 'spectrogram_jerk.png'))
        plt.show()   
        trm_amp_list= psd[peaks].tolist()
        if len(trm_amp_list)>0:
            trm_amp_list.remove(max(psd[peaks]))
            tremor_amplitude.append(np.mean(trm_amp_list))
        
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_jerk_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData2\\Spectrum_Jerk'
create_folder(path_jerk_spectrum_plots)
healthy_amplitude = []
peak_list_healthy=[]
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        heal_jerk, heal_magnitude, heal_hrz_jerk, heal_vert_jerk, heal_TimeStamp_difference , heal_magnitude_jerk, heal_magnitude_horz_jerk, heal_magnitude_vert_jerk, heal_timestamps =find_jerk(df) 
        df_jerk = pd.DataFrame(columns=['TimeStamp', 'Jerk'])
        df_jerk['Jerk'] = heal_magnitude
        data=df_jerk['Jerk'].tolist()
        Fs= 133
        win = 4* Fs
        freqs, psd = signal.welch(data, Fs, nperseg=win)
        peaks, _ = find_peaks(psd, height=np.amax(psd)*0.05)
        peak_list_healthy.append(peaks)
        plt.plot(psd)
        plt.plot(peaks, psd[peaks], "x")
        plt.plot(np.zeros_like(psd), "--", color="gray")
        plt.savefig(os.path.join(path_jerk_spectrum_plots,os.path.basename(file) + '_spectrum_jerk.png'))
        plt.show()   
        heal_amp_list= psd[peaks].tolist()
        if len(heal_amp_list)>0:
            heal_amp_list.remove(max(psd[peaks]))
            healthy_amplitude.append(np.mean(heal_amp_list))
        
#%%
import matplotlib.pyplot as plt
plt.plot(Frac_Healthy,'g*', Frac_Tremor, 'ro')
plt.show()
#%%
