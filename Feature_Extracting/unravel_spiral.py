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
import scipy.signal as sig
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange
from scipy.signal import find_peaks
from random import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
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
    lookahead = 100
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
        fig = sns.lineplot(df['Radius'], df['TimeStamp'])
        fig.set(xlabel='TimeStamp', ylabel='Radius')
        path_folder = os.path.join(data_path, "plots_unravelled_spirals")
        create_folder(path_folder)
        fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_unravelling.png'))
        plt.title("Unravelled Spiral(Healthy)")
        """
        plt.xlim(0, 1.4)
        plt.ylim(0, 800)
        """
        plt.show()
#%%
def calculate_sampling_frequency(path_file):
    df = read_data_file(path_file)
    radius, theta = unravel_spiral(df)
    df_theta = pd.DataFrame(columns=['TimeStamp', 'Theta'])
    df_theta['Theta'] = theta
    df_list1= df_theta['Theta'].tolist()
    number_of_samples= len(df_list1)
    df_list2= df['TimeStamp'].tolist()
    total_time= (df_list2[len(df_list2)-1]-df_list2[0]) 
    sampling_frequency= number_of_samples/total_time
    return sampling_frequency
#%%
from entropy import *
import numpy as np
import hfda
from random import sample 
import statistics
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Frac_Tremor= []
std_Tremor = []
perm_en_tremor= []
spectral_en_tremor= []
svd_en_tremor= []
app_en_tremor= []
sample_en_tremor= []
detrend_fluc_tremor= []
Tremor_smooth= []
sarima_tremor = []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        """
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        df_polar_coordinate['Radius'] = radius
        Tremor_unravel= df_polar_coordinate['Radius'].tolist()
        D = hfda.measure(np.array(Tremor_unravel), 6)
        Frac_Tremor.append(D)
        std_Tremor.append((statistics.stdev(Tremor_unravel))) 
        perm_en_tremor.append(perm_entropy(Tremor_unravel, order=3, normalize=True))                          # Permutation entropy
        spectral_en_tremor.append(spectral_entropy(Tremor_unravel, 133, nperseg= 64, method='welch', normalize=True))      # Spectral entropy
        svd_en_tremor.append(svd_entropy(Tremor_unravel, order=3, delay=1, normalize=True))                   # Singular value decomposition entropy
        app_en_tremor.append(app_entropy(Tremor_unravel, order=2, metric='chebyshev'))                        # Approximate entropy
        sample_en_tremor.append(sample_entropy(Tremor_unravel, order=2, metric='chebyshev'))                  # Sample entropy
        model = ExponentialSmoothing(Tremor_unravel)
        model_fit = model.fit()
        yhat = model_fit.predict(len(Tremor_unravel), len(Tremor_unravel))
        model2 = SARIMAX(Tremor_unravel, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
        model_fit2 = model2.fit(disp=False)
        yhat2 = model_fit2.predict(len(Tremor_unravel), len(Tremor_unravel))
        sarima_tremor.append(yhat2)
        Tremor_smooth.append(yhat)
        """
        plt.plot(radius,'r')
        plt.title("Residue from DCT Coefficients (Lines:Healthy)")
        plt.show()
#%%
from entropy import *
import numpy as np
import hfda
from random import sample 
import statistics
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Frac_Healthy= []
std_healthy = []
perm_en_healthy= []
spectral_en_healthy= []
svd_en_healthy= []
app_en_healthy= []
sample_en_healthy= []
detrend_fluc_healthy= []
Healthy_smooth = []
sarima_healthy = []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        """
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        df_polar_coordinate['Radius'] = radius
        Healthy_unravel = df_polar_coordinate['Radius'].tolist()
        D = hfda.measure(np.array(Healthy_unravel), 6)
        Frac_Healthy.append(D)
        std_healthy.append((statistics.stdev(Healthy_unravel)))
        perm_en_healthy.append(perm_entropy(Healthy_unravel, order=3, normalize=True))                                       # Permutation entropy
        spectral_en_healthy.append(spectral_entropy(Healthy_unravel, 133, nperseg= 64, method='welch', normalize=True))      # Spectral entropy
        svd_en_healthy.append(svd_entropy(Healthy_unravel, order=3, delay=1, normalize=True))                   # Singular value decomposition entropy
        app_en_healthy.append(app_entropy(Healthy_unravel, order=2, metric='chebyshev'))                        # Approximate entropy
        sample_en_healthy.append(sample_entropy(Healthy_unravel, order=2, metric='chebyshev'))# Sample entropy
        model = ExponentialSmoothing(Healthy_unravel)
        model_fit = model.fit()
        yhat = model_fit.predict(len(Healthy_unravel), len(Healthy_unravel))
        model2 = SARIMAX(Healthy_unravel, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
        model_fit2 = model2.fit(disp=False)
        yhat2 = model_fit2.predict(len(Healthy_unravel), len(Healthy_unravel))
        sarima_healthy.append(yhat2)
        Healthy_smooth.append(yhat)
        plot_unravelled_spiral(df_polar_coordinate, file)
        """
        plt.plot(radius,'g')
        plt.title("Residue from DCT Coefficients (Lines:Healthy)")
        plt.show()
#%%

base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_unravelled_spiral_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData2\\FFT_Unravelled_Spiral'
create_folder(path_unravelled_spiral_plots)
for file in file_list: 
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        df_polar_coordinate['Radius'] = radius
        X = df_polar_coordinate['Radius'].values
        n = len(X) # length of the signal
        k = arange(n)
        Fs = 133;  # sampling rate
        T = n/Fs
        frq = k/T # two sides frequency range
        frq = frq[range(int(n/2))] # one side frequency range
        Y = fft(X)/n # fft computing and normalization
        Y = Y[range(int(n/2))]
        plt.plot(frq,abs(Y),linestyle='-', color='blue') # plotting the spectrum
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.savefig(os.path.join(path_unravelled_spiral_plots,os.path.basename(file) + '_FFT_Unravell_spiral.png'))
        plt.show()
        
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
path_unravelled_spiral_plots = 'D:\\Autumn 2019\\Research\\Feature_Extracting\\TabletData2\\FFT_Unravelled_Spiral'
create_folder(path_unravelled_spiral_plots)
frequency = []
amplitude = []
for file in file_list: 
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        df_polar_coordinate['Radius'] = radius
        X = df_polar_coordinate['Radius'].values
        n = len(X) # length of the signal
        k = arange(n)
        Fs = 133;  # sampling rate
        T = n/Fs
        frq = k/T # two sides frequency range
        frq = frq[range(int(n/2))] # one side frequency range
        Y = fft(X)/n # fft computing and normalization
        Y = Y[range(int(n/2))]
        frequency.append(frq)
        amplitude.append(abs(Y))
        print(frequency)
        print(amplitude)
        plt.plot(frq,abs(Y),linestyle='-', color='blue') # plotting the spectrum
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.savefig(os.path.join(path_unravelled_spiral_plots,os.path.basename(file) + '_FFT_Unravell_spiral.png'))
        plt.show()

#%%
import matplotlib.pyplot as plt
"""
plt.plot(Healthy_smooth,'g*', Tremor_smooth, 'ro')
plt.xticks([])
plt.title('Exponential Smoothing ')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(Healthy_smooth)), Healthy_smooth, 'g*')
plt.plot(np.zeros(np.shape(Tremor_smooth)), Tremor_smooth, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Exponential Smoothing')
plt.show()
# %%
import matplotlib.pyplot as plt
"""
plt.plot(sarima_healthy,'g*', sarima_tremor, 'ro')
plt.xticks([])
plt.title('Seasonal Autoregressive Integrated Moving Average')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(sarima_healthy)), sarima_healthy, 'g*')
plt.plot(np.zeros(np.shape(sarima_tremor)), sarima_tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Seasonal Autoregressive Integrated Moving Average')
plt.show()
# %%
from scipy import signal
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
df = read_data_file(file_list[15])
radius, theta = unravel_spiral(df)
df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
df_polar_coordinate['Angle'] = theta
df_polar_coordinate['Radius'] = radius
Tremor_unravel= df_polar_coordinate['Angle'].tolist()
b= np.asarray(Tremor_unravel)
F= calculate_sampling_frequency(file_list[15])
f, t, Sxx = signal.spectrogram(b, F, nperseg=8)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("Spectrogram of Angle (Unravelled Spiral)")
plt.show()
# %%
