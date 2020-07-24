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
    return acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps

#%%
def calculate_sampling_frequency2(path_file):
    df = read_data_file(path_file)
    acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps = acceleration_extraction(df)
    df_acceleration = pd.DataFrame(columns=['TimeStamp', 'Acceleration'])
    df_acceleration['Acceleration'] = acceleration_magnitude
    df_list1= df_acceleration['Acceleration'].tolist()
    number_of_samples= len(df_list1)
    df_list2= df['TimeStamp'].tolist()
    total_time= (df_list2[len(df_list2)-1]-df_list2[0]) 
    sampling_frequency= number_of_samples/total_time
    return sampling_frequency
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting" 
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Tremor_Acceleration= []
def plot_Acceleration(df, path_file):
    fig = sns.tsplot(df['Acceleration'], df['TimeStamp'])
    fig.set(xlabel='Time', ylabel='Acceleration', ylim= (0, 100000), title= 'Acceleration: Tremor' )
    path_folder = os.path.join(data_path, "plots_Acceleration")
    create_folder(path_folder)
    fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_Acceleration.png'))
    plt.show()
    
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps = acceleration_extraction(df)
        df_acceleration = pd.DataFrame(columns=['TimeStamp', 'Acceleration'])
        df_acceleration['TimeStamp'] = timestamps
        df_acceleration['Acceleration'] = acceleration_magnitude
        Tremor_Acceleration.append(np.mean(acceleration_magnitude))
        #plot_Acceleration(df_acceleration, file)
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting" 
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Healthy_Acceleration= []
def plot_Acceleration(df, path_file):
    fig = sns.tsplot(df['Acceleration'], df['TimeStamp'])
    fig.set(xlabel='Time', ylabel='Acceleration', ylim= (0, 100000), title= 'Acceleration: Healthy')
    path_folder = os.path.join(data_path, "plots_Acceleration")
    create_folder(path_folder)
    fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_Acceleration.png'))
    plt.show()
    
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps = acceleration_extraction(df)
        df_acceleration = pd.DataFrame(columns=['TimeStamp', 'Acceleration'])
        df_acceleration['TimeStamp'] = timestamps
        df_acceleration['Acceleration'] = acceleration_magnitude
        Healthy_Acceleration.append(np.mean(acceleration_magnitude))
        #plot_Acceleration(df_acceleration, file)
#%%
import matplotlib.pyplot as plt
"""
plt.plot(Healthy_Acceleration,'g*', Tremor_Acceleration, 'ro')
plt.xticks([])
plt.title('Acceleration (Lines)')
plt.show()
"""

plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(Healthy_Acceleration)), Healthy_Acceleration, 'g*')
plt.plot(np.zeros(np.shape(Tremor_Acceleration)), Tremor_Acceleration, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Acceleration (Spirals)')
plt.show()
#%%
from PyEMD import EMD
import numpy as np
import pylab as plt
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
data_path=os.path.join(base_folder, 'TabletData2')
healthy_amplitude = []
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_acceleration_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData1\\acceleration_emd_spectrum_plot'
create_folder(path_acceleration_spectrum_plots)
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps = acceleration_extraction(df)
        df_acceleration = pd.DataFrame(columns=['TimeStamp', 'Acceleration'])
        df_acceleration['Acceleration'] = acceleration_magnitude
        s=df_acceleration['Acceleration'].values
        t = timestamps
        IMF = EMD().emd(s,t)
        N = IMF.shape[0]+1
        #plt.subplot(N,1,1)
        #plt.plot(t, s, 'r')
        #for n, imf in enumerate(IMF):
            #plt.subplot(N,1,n+2)
        #plt.plot(t, IMF[0], 'g')
        #plt.show()
        Fs= 133
        win = 4* Fs
        freqs, psd = signal.welch(IMF[3], Fs, nperseg=win)
        '''
        fig = plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's Velocity Spectrum (Healthy)")
        plt.xlim([0, freqs.max()])
        plt.savefig(os.path.join(path_acceleration_spectrum_plots,os.path.basename(file) + '_AccelerationSpectrogram.png'))
        plt.show() 
        '''
        peaks, _ = find_peaks(psd, height=np.amax(psd)*0.05)
        plt.plot(psd)
        plt.plot(peaks, psd[peaks], "x")
        plt.plot(np.zeros_like(psd), "--", color="gray")
        plt.savefig(os.path.join(path_acceleration_spectrum_plots,os.path.basename(file) + '_AccelerationSpectrogram.png'))
        plt.show()   
        hel_amp_list= psd[peaks].tolist()
        healthy_amplitude.append(np.mean(hel_amp_list))
#%%
from PyEMD import EMD
import numpy as np
import pylab as plt
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
data_path=os.path.join(base_folder, 'TabletData1')
tremor_amplitude = []
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_acceleration_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData1\\acceleration_emd_spectrum_plot'
create_folder(path_acceleration_spectrum_plots)
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        acceleration, horizontal_acceleration, vertical_acceleration, horizontal_acceleration_magnitude, vertical_acceleration_magnitude, acceleration_magnitude, timestamps = acceleration_extraction(df)
        df_acceleration = pd.DataFrame(columns=['TimeStamp', 'Acceleration'])
        df_acceleration['Acceleration'] = acceleration_magnitude
        s=df_acceleration['Acceleration'].values
        t = timestamps
        IMF = EMD().emd(s,t)
        N = IMF.shape[0]+1
        #plt.subplot(N,1,1)
        #plt.plot(t, s, 'r')
        #for n, imf in enumerate(IMF):
            #plt.subplot(N,1,n+2)
        #plt.plot(t, IMF[0], 'g')
        #plt.show()
        Fs= 133
        win = 4* Fs
        freqs, psd = signal.welch(IMF[3], Fs, nperseg=win)
        '''
        fig = plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's Velocity Spectrum (Healthy)")
        plt.xlim([0, freqs.max()])
        plt.savefig(os.path.join(path_acceleration_spectrum_plots,os.path.basename(file) + '_AccelerationSpectrogram.png'))
        plt.show() 
        '''
        peaks, _ = find_peaks(psd, height=np.amax(psd)*0.05)
        plt.plot(psd)
        plt.plot(peaks, psd[peaks], "x")
        plt.plot(np.zeros_like(psd), "--", color="gray")
        plt.savefig(os.path.join(path_acceleration_spectrum_plots,os.path.basename(file) + '_AccelerationSpectrogram.png'))
        plt.show()   
        trm_amp_list= psd[peaks].tolist()
        tremor_amplitude.append(np.mean(trm_amp_list))


#%%
def mean_acceleration_extraction(df):
    Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
    acceleration = []
    horizontal_acceleration =  []
    vertical_acceleration = []
    acceleration_magnitude = []
    horizontal_acceleration_magnitude = []
    vertical_acceleration_magnitude = []
    timestamps = []
    t= 0
    lookahead=5
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
    return mean_acceleration_magnitude, mean_horizontal_acceleration_magnitude, mean_vertical_acceleration_magnitude

#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
healthy_acc = []
healthy_hor= []
healthy_ver= []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        mean_acceleration_magnitude, mean_horizontal_acceleration_magnitude, mean_vertical_acceleration_magnitude= mean_acceleration_extraction(df)
        healthy_acc.append(mean_acceleration_magnitude)
        healthy_hor.append(mean_horizontal_acceleration_magnitude)
        healthy_ver.append(mean_vertical_acceleration_magnitude)

# %%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
tremor_acc = []
tremor_hor= []
tremor_ver= []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        mean_acceleration_magnitude, mean_horizontal_acceleration_magnitude, mean_vertical_acceleration_magnitude= mean_acceleration_extraction(df)
        tremor_acc.append(mean_acceleration_magnitude)
        tremor_hor.append(mean_horizontal_acceleration_magnitude)
        tremor_ver.append(mean_vertical_acceleration_magnitude)

#%%
import matplotlib.pyplot as plt
plt.plot(healthy_hor,'g*', tremor_hor, 'ro')
plt.title('Acceleration')
plt.show()


# %%
