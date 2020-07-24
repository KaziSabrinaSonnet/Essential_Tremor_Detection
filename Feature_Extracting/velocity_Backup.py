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
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Healthy_Velocity = []
def plot_velocity(df, path_file):
        fig = sns.tsplot(df['Velocity'], df['TimeStamp'])
        fig.set(xlabel='Time', ylabel='Velocity', ylim= (0, 2000))
        path_folder = os.path.join(data_path, "plots_velocity")
        create_folder(path_folder)
        fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_velocity.png'))
        plt.show()

for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
        df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
        df_velocity['TimeStamp'] = timestamps
        df_velocity['Velocity'] = velocity_magnitude
        Healthy_Velocity.append(np.mean(velocity_magnitude))
        plot_velocity(df_velocity, file)
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'StimOn')
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Tremor_Velocity = []
def plot_velocity(df, path_file):
        fig = sns.tsplot(df['Velocity'], df['TimeStamp'])
        fig.set(xlabel='Time', ylabel='Velocity', ylim= (0, 1500))
        path_folder = os.path.join(data_path, "plots_velocity")
        create_folder(path_folder)
        fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_velocity.png'))
        plt.title("Velocity(Tremor)")
        plt.show()

for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
        df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
        df_velocity['TimeStamp'] = timestamps
        df_velocity['Velocity'] = velocity_magnitude
        Tremor_Velocity.append(np.mean(velocity_magnitude))
        print(np.mean(velocity_magnitude))
        plot_velocity(df_velocity, file)
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Healthy_Velocity = []
def plot_velocity(df, path_file):
        fig = sns.tsplot(df['Velocity'], df['TimeStamp'])
        fig.set(xlabel='Time', ylabel='Velocity', ylim= (0, 1500))
        path_folder = os.path.join(data_path, "plots_velocity")
        create_folder(path_folder)
        fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_velocity.png'))
        plt.title("Velocity(Healthy)")
        plt.show()

for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
        df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
        df_velocity['TimeStamp'] = timestamps
        df_velocity['Velocity'] = velocity_magnitude
        Healthy_Velocity.append(np.mean(velocity_magnitude))
        plot_velocity(df_velocity, file)

#%%
import matplotlib.pyplot as plt
"""
plt.plot(Healthy_Velocity,'g*', Tremor_Velocity, 'ro')
plt.title('Velocity')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(Healthy_Velocity)), Healthy_Velocity, 'g*')
plt.plot(np.zeros(np.shape(Tremor_Velocity)), Tremor_Velocity, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Velocity (Lines)')
plt.show()
#%%
def calculate_sampling_frequency(path_file):
    df = read_data_file(path_file)
    Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
    df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
    df_velocity['Velocity'] = velocity_magnitude
    df_list1= df_velocity['Velocity'].tolist()
    number_of_samples= len(df_list1)
    df_list2= df['TimeStamp'].tolist()
    total_time= (df_list2[len(df_list2)-1]-df_list2[0]) 
    sampling_frequency= number_of_samples/total_time
    return sampling_frequency

#%%
path_velocity_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData1\\plots_velocity_in_frequency_domain'
create_folder(path_velocity_spectrum_plots)
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
        df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
        df_velocity['Velocity'] = velocity_magnitude
        data=df_velocity['Velocity'].tolist()
        Fs= calculate_sampling_frequency(file)
        win = 4* Fs
        freqs, psd = signal.welch(data, Fs, nperseg=win)
        fig = plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's Velocity Spectrum (Healthy)")
        plt.xlim([0, freqs.max()])
        plt.ylim(0, 1)
        plt.savefig(os.path.join(path_velocity_spectrum_plots,os.path.basename(file) + '_velocity_spectrum.png'))
        plt.show() 
    

#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_velocity_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData1\\Local_maxima'
create_folder(path_velocity_spectrum_plots)
tremor_amplitude = []
peak_list_tremor=[]
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
        df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
        df_velocity['Velocity'] = velocity_magnitude
        data=df_velocity['Velocity'].tolist()
        Fs= calculate_sampling_frequency(file)
        win = 4* Fs
        freqs, psd = signal.welch(data, Fs, nperseg=win)
        peaks, _ = find_peaks(psd, height=np.amax(psd)*0.05)
        peak_list_tremor.append(np.mean(peaks))
        trm_amp_list= peaks.tolist()
        tremor_amplitude.append((np.mean(trm_amp_list)))
        plt.plot(psd)
        plt.plot(peaks, psd[peaks], "x")
        plt.plot(np.zeros_like(psd), "--", color="gray")
plt.yticks([])
plt.xlim(right= 65)
plt.xlabel('Frequency(Hz)')
plt.title("Welch's Line Velocity Spectrum (Tremor)")
plt.show() 
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting" 
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_velocity_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData2\\Local_maxima'
create_folder(path_velocity_spectrum_plots)
healthy_amplitude = []
peak_list_healthy=[]
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
        df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
        df_velocity['Velocity'] = velocity_magnitude
        data=df_velocity['Velocity'].tolist()
        Fs= calculate_sampling_frequency(file)
        win = 4* Fs
        freqs, psd = signal.welch(data, Fs, nperseg=win)
        peaks, _ = find_peaks(psd, height=np.amax(psd)*0.05)
        peak_list_healthy.append(np.mean(peaks))
        plt.plot(psd)
        plt.plot(peaks, psd[peaks], "x")
        plt.plot(np.zeros_like(psd), "--", color="gray")
        #plt.savefig(os.path.join(path_velocity_spectrum_plots,os.path.basename(file) + '_local_maxima.png'))
        hel_amp_list= peaks.tolist()
        healthy_amplitude.append((np.mean(hel_amp_list)))
plt.yticks([])
plt.xlabel('Frequency(Hz)')
plt.xlim(right=65)
plt.title("Welch's Line Velocity Spectrum (Healthy)")
plt.show()  
        
#%%
import math
tremor_amplitude = [0 if math.isnan(x) else x for x in tremor_amplitude]
healthy_amplitude = [0 if math.isnan(x) else x for x in healthy_amplitude]       
#%%

    """
    plt.plot(peak_list_healthy[i],'g*', peak_list_tremor[i], 'ro')
    plt.xlabel('index')
    plt.ylabel('frequency at peaks')
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.savefig(os.path.join(path_velocity_spectrum_plots,os.path.basename(file_list[i]) + '_compare.png'))
    plt.show()
    """
#peak_list_healthy.remove(32.36363636363637)
peak_list_healthy.remove(38.705882352941174)
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(peak_list_healthy)), peak_list_healthy, 'g*')
plt.plot(np.zeros(np.shape(peak_list_tremor)), peak_list_tremor, 'ro')
plt.tick_params(
axis='x',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom=False,      # ticks along the bottom edge are off
top=False,         # ticks along the top edge are off
labelbottom=False) # labels along the bottom edge are off
plt.title('Number of Detected Peaks in Velocity Spectrogram (Spirals)')
plt.show()
# %%
for item in peak_list_healthy:
    int(item)
# %%
for item in peak_list_tremor:
    k.append(float(item))
#%%
def mean_velocity_extraction(df):
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
    return mean_velocity_magnitude, mean_vertical_velocity_magnitude, mean_horizontal_velocity_magnitude 

#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
healthy_vel = []
healthy_horvel= []
healthy_vervel= []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        mean_velocity_magnitude, mean_vertical_velocity_magnitude, mean_horizontal_velocity_magnitude  = mean_velocity_extraction(df)
        healthy_vel.append(mean_velocity_magnitude)
        healthy_horvel.append(mean_horizontal_velocity_magnitude)
        healthy_vervel.append(mean_vertical_velocity_magnitude)

#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
    
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
tremor_vel = []
tremor_horvel= []
tremor_vervel= []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        mean_velocity_magnitude, mean_vertical_velocity_magnitude, mean_horizontal_velocity_magnitude  = mean_velocity_extraction(df)
        tremor_vel.append(mean_velocity_magnitude)
        tremor_horvel.append(mean_horizontal_velocity_magnitude)
        tremor_vervel.append(mean_vertical_velocity_magnitude)

#%%
import math
tremor_vel = [0 if math.isnan(x) else x for x in tremor_vel]
healthy_vel = [0 if math.isnan(x) else x for x in healthy_vel]    
#%%

base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting" 
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_velocity_spectrum_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData2\\Local_maxima'
create_folder(path_velocity_spectrum_plots)
healthy_amplitude = []
peak_list_healthy=[]
df= read_data_file(file_list[14])
Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
df_velocity['Velocity'] = velocity_magnitude
data=df_velocity['Velocity'].tolist()
Fs= calculate_sampling_frequency(file_list[11])
win = 4* Fs
freqs, psd = signal.welch(data, Fs, nperseg=win)
peaks, _ = find_peaks(psd, height=np.amax(psd)*0.05)
peak_list_healthy.append(peaks)
a= psd.tolist()
norm = [i/sum(a) for i in a]
b= psd[peaks].tolist()
norm1 = [i/sum(b) for i in b]
c= peaks.tolist()
norm2 = [i/sum(c) for i in c]
plt.plot(norm)
plt.plot(norm2, norm1, "x")
plt.plot(np.zeros_like(norm), "--", color="gray")


#plt.savefig(os.path.join(path_velocity_spectrum_plots,os.path.basename(file) + '_local_maxima.png'))
plt.ylim(0, 1)
plt.title("Welch's Velocity Spectrum (Healthy)") 
plt.show()  
hel_amp_list= peaks.tolist()
healthy_amplitude.append(int(np.mean(hel_amp_list)))
#%%
from scipy import signal
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
df= read_data_file(file_list[16])
Velocity, horizontal_velocity, vertical_velocity, velocity_magnitude, TimeStamp_difference, horizontal_velocity_magnitude, vertical_velocity_magnitude, timestamps = velocity_extraction(df)
df_velocity = pd.DataFrame(columns=['TimeStamp', 'Velocity'])
df_velocity['Velocity'] = velocity_magnitude
data=df_velocity['Velocity'].tolist()
b= np.asarray(data)
Fs= calculate_sampling_frequency(file_list[12])
f, t, Sxx = signal.spectrogram(b, 121, nperseg=16)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# %%
