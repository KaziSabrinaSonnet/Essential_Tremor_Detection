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
from entropy import *
import numpy as np
import hfda
c
import statistics
from statsmodels.tsa.ar_model import AR
from random import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
from scipy.fftpack import ifft, idct
from scipy.fftpack import fft, dct
def get_inverse_cosine_transform_X(df): 
    dis_cos_ts = []
    lookahead = 10
    t = 0
    for i in range(len(df)-lookahead):
        if t+lookahead <= len(df)-1:
            dis_cos_ts.append(df['XEnd'][t])
            t= t+lookahead
    return dis_cos_ts

def do_inverse_cosine_transform_X(df): 
    dct_list_X= (dct(np.array(get_inverse_cosine_transform_X(df)), 1)).tolist()
    idct_list_X= (idct(np.array(dct_list_X), 1)).tolist()
    return idct_list_X
#%%
from scipy.fftpack import ifft, idct
def get_inverse_cosine_transform_Y(df): 
    dis_cos_ts = []
    lookahead = 10
    t = 0
    for i in range(len(df)-lookahead):
        if t+lookahead <= len(df)-1:
            dis_cos_ts.append(df['YEnd'][t])
            t= t+lookahead
    return dis_cos_ts

def do_inverse_cosine_transform_Y(df): 
    dct_list_Y= (dct(np.array(get_inverse_cosine_transform_Y(df)), 1)).tolist()
    idct_list_Y= (idct(np.array(dct_list_Y), 1)).tolist()
    return idct_list_Y

#%%
#Single
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Frac_Healthy =[]
std_healthy = []
perm_en_healthy= []
spectral_en_healthy= []
svd_en_healthy= []
app_en_healthy= []
sample_en_healthy= []
detrend_fluc_healthy= []
healthy_autoregression = []
healthy_moving_average = []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        Healthy_idct = []
        Healthy_idct_X = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_X(df), do_inverse_cosine_transform_X(df))]
        Healthy_idct_Y = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_Y(df), do_inverse_cosine_transform_Y(df))]
        t= 0
        idct_square = 0
        for i in range(0, len(Healthy_idct_X)-1):
            idct_square = (math.sqrt(Healthy_idct_X[t]**2+Healthy_idct_Y[t]**2))
            Healthy_idct.append(idct_square)
            t = t+1
        D = hfda.measure(np.array(Healthy_idct), 6)
        Frac_Healthy.append(D)
        std_healthy.append((statistics.stdev(Healthy_idct)))
        perm_en_healthy.append(perm_entropy(Healthy_idct, order=2, normalize=True))                                       # Permutation entropy
        spectral_en_healthy.append(spectral_entropy(Healthy_idct, 133, nperseg= 64, method='welch', normalize=True))      # Spectral entropy
        svd_en_healthy.append(svd_entropy(Healthy_idct, order=2, delay=1, normalize=True))                   # Singular value decomposition entropy
        app_en_healthy.append(app_entropy(Healthy_idct, order=2, metric='chebyshev'))                        # Approximate entropy
        sample_en_healthy.append(sample_entropy(Healthy_idct, order=2, metric='chebyshev'))
        #detrend_fluc_healthy.append(detrended_fluctuation(Healthy_idct))                 # Sample entropy
        model = AR(Healthy_idct)
        model_fit = model.fit()
        yhat = model_fit.predict(len(Healthy_idct), len(Healthy_idct))
        healthy_autoregression.append(yhat)
        model1 = ExponentialSmoothing(Healthy_idct)
        model_fit1 = model1.fit()
        yhat1 = model_fit1.predict(len(Healthy_idct), len(Healthy_idct))
        healthy_moving_average.append(yhat1)
        plt.xlabel ('Points')
        plt.ylabel('Residue')
        plt.ylim (0, 800000)
        plt.plot(sample(Healthy_idct, 30),'g')
        plt.title("Residue from DCT Coefficients (Lines:Healthy)")
        plt.show()
        
            
#%%
#Single
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
tremor_autoregression= []
tremor_moving_average= []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        Tremor_idct = []
        Tremor_idct_X = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_X(df), do_inverse_cosine_transform_X(df))]
        Tremor_idct_Y = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_Y(df), do_inverse_cosine_transform_Y(df))]
        t= 0
        idct_square = 0
        for i in range(0, len(Tremor_idct_X)-1):
            idct_square = (math.sqrt(Tremor_idct_X[t]**2+ Tremor_idct_Y[t]**2))
            Tremor_idct.append(idct_square)
            t = t+1
        D = hfda.measure(np.array(Tremor_idct), 6)
        Frac_Tremor.append(D)
        std_Tremor.append((statistics.stdev(Tremor_idct)))
        #detrend_fluc_tremor.append(detrended_fluctuation(Tremor_idct))  
        perm_en_tremor.append(perm_entropy(Tremor_idct, order=2, normalize=True))                          # Permutation entropy
        spectral_en_tremor.append(spectral_entropy(Tremor_idct, 133, nperseg= 64, method='welch', normalize=True))      # Spectral entropy
        svd_en_tremor.append(svd_entropy(Tremor_idct, order=2, delay=1, normalize=True))                   # Singular value decomposition entropy
        app_en_tremor.append(app_entropy(Tremor_idct, order=2, metric='chebyshev'))                        # Approximate entropy
        sample_en_tremor.append(sample_entropy(Tremor_idct, order=2, metric='chebyshev'))                  # Sample entropy
        model = AR(Tremor_idct)
        model_fit = model.fit()
        yhat = model_fit.predict(len(Tremor_idct), len(Tremor_idct))
        tremor_autoregression.append(yhat)
        model1 = ExponentialSmoothing(Tremor_idct)
        model_fit1 = model1.fit()
        yhat1 = model_fit1.predict(len(Tremor_idct), len(Tremor_idct))
        tremor_moving_average.append(yhat1)
        plt.xlabel ('Points')
        plt.ylabel('Residue')
        plt.ylim (0, 800000)
        plt.title("Residue from DCT Coefficients (Lines:Tremor)")
        plt.plot(sample(Tremor_idct, 30),'r')
        plt.show()           
        

#%%
#Spectrogram 
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
healthy_psd= []
for file in file_list:
    if os.path.isfile(file):
        df= read_data_file(file)
        Healthy_idct = []
        Healthy_idct_X = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_X(df), do_inverse_cosine_transform_X(df))]
        Healthy_idct_Y = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_Y(df), do_inverse_cosine_transform_Y(df))]
        t= 0
        idct_square = 0
        for i in range(0, len(Healthy_idct_X)-1):
            idct_square = (math.sqrt(Healthy_idct_X[t]**2+Healthy_idct_Y[t]**2))
            t = t+1
            Healthy_idct.append(idct_square)
        data= Healthy_idct
        number_of_samples= len(data)
        df_list2= df['TimeStamp'].tolist()
        total_time= (df_list2[len(df_list2)-1]-df_list2[0]) 
        Fs= number_of_samples/total_time 
        win = 4* Fs
        freqs, psd = signal.welch(data, Fs, nperseg=win)
        healthy_psd.append(psd)
        fig = plt.figure(figsize=(8, 7))
        plt.plot(freqs, psd, lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.ylim([0, 4*10**10])
        plt.title("Welch's Velocity Spectrum (Healthy)")
        plt.xlim([0, freqs.max()])
        plt.show() 
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
tremor_psd = []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        Tremor_idct = []
        Tremor_idct_X = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_X(df), do_inverse_cosine_transform_X(df))]
        Tremor_idct_Y = [a_i - b_i for a_i, b_i in zip(get_inverse_cosine_transform_Y(df), do_inverse_cosine_transform_Y(df))]
        t= 0
        idct_square = 0
        for i in range(0, len(Tremor_idct_X)-1):
            idct_square = (math.sqrt(Tremor_idct_X[t]**2+ Tremor_idct_Y[t]**2))
            Tremor_idct.append(idct_square)
            t = t+1
        data= Tremor_idct 
        number_of_samples= len(data)
        df_list2= df['TimeStamp'].tolist()
        total_time= (df_list2[len(df_list2)-1]-df_list2[0]) 
        Fs= number_of_samples/total_time 
        win = 4* Fs
        freqs, psd = signal.welch(data, Fs, nperseg=win)
        tremor_psd.append(psd)
        fig = plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.ylim([0, 4*10**10])
        plt.title("Welch's Velocity Spectrum (Tremor)")
        plt.xlim([0, freqs.max()])
        plt.show() 

#%%
import matplotlib.pyplot as plt
"""
plt.plot(std_healthy,'g*', std_Tremor, 'ro')
plt.xticks([])
plt.title('Standard Deviation (Spirals)')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(std_healthy)), std_healthy, 'g*')
plt.plot(np.zeros(np.shape(std_Tremor)), std_Tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Standard Deviation (Spirals)')
plt.show()

#%%
import matplotlib.pyplot as plt
"""
plt.plot(Frac_Healthy,'g*', Frac_Tremor, 'ro')
plt.xticks([])
plt.title('Fractral Dimension')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(Frac_Healthy)), Frac_Healthy, 'g*')
plt.plot(np.zeros(np.shape(Frac_Tremor)), Frac_Tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Fractal Deminsion (Spirals)')
plt.show()


#%%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from matplotlib.patches import Circle
"""
plt.plot(app_en_healthy,'g*', app_en_tremor, 'ro')
plt.xticks([])
plt.title('Approximate Entropy (Lines)')
plt.show()
"""

plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(app_en_healthy)), app_en_healthy, 'g*')
plt.plot(np.zeros(np.shape(app_en_tremor)), app_en_tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Approximate Entropy (Spirals)')
legend_elements = [Line2D([0], [0], marker='*', color='w', label='Healthy',
                          markerfacecolor='g', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Tremor',
                          markerfacecolor='r', markersize=7)]

plt.legend(handles=legend_elements, loc='upper left')

#plt.legend((app_en_healthy, app_en_tremor),('Healthy', 'Tremor'), numpoints=1, loc='upper left', ncol=3, fontsize=8)
plt.show()

#%%
import matplotlib.pyplot as plt
"""
plt.plot(svd_en_healthy,'g*', svd_en_tremor, 'ro')
plt.xticks([])
plt.title('Singular Value Decomposition')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(svd_en_healthy)), svd_en_healthy, 'g*')
plt.plot(np.zeros(np.shape(svd_en_tremor)), svd_en_tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Singular Value Decomposition (Spirals)')
plt.show()



#%%
import matplotlib.pyplot as plt
"""
plt.plot(sample_en_healthy,'g*', sample_en_tremor, 'ro')
plt.xticks([])
plt.title('Sample Entropy')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(sample_en_healthy)), sample_en_healthy, 'g*')
plt.plot(np.zeros(np.shape(sample_en_tremor)), sample_en_tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Sample Entropy (Spirals)')
plt.show()

#%%
import matplotlib.pyplot as plt
"""
plt.plot(spectral_en_healthy,'g*', spectral_en_tremor, 'ro')
plt.xticks([])
plt.title('Spectral Entropy (Lines)')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(spectral_en_healthy)), spectral_en_healthy, 'g*')
plt.plot(np.zeros(np.shape(spectral_en_tremor)), spectral_en_tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Spectral Entropy (Spirals)')
plt.show()
#%%
import matplotlib.pyplot as plt
"""
plt.plot(perm_en_healthy,'g*', perm_en_tremor, 'ro')
plt.xticks([])
plt.title('Permutation Entropy')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(perm_en_healthy)), perm_en_healthy, 'g*')
plt.plot(np.zeros(np.shape(perm_en_tremor)), perm_en_tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Permutation Entropy (Spirals)')
plt.show()
#%%
import matplotlib.pyplot as plt
"""
plt.plot(detrend_fluc_healthy,'g*', detrend_fluc_tremor, 'ro')
plt.xticks([])
plt.title('Detrend Fluctuation')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(detrend_fluc_healthy)), detrend_fluc_healthy, 'g*')
plt.plot(np.zeros(np.shape(detrend_fluc_tremor)), detrend_fluc_tremor, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Detrend Fluctuation')
plt.show()
# %%
import matplotlib.pyplot as plt
"""
plt.plot(healthy_autoregression,'g*', tremor_autoregression, 'ro')
plt.xticks([])
plt.title('Autoregression (Lines)')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(healthy_autoregression)), healthy_autoregression, 'g*')
plt.plot(np.zeros(np.shape(tremor_autoregression)), tremor_autoregression, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title('Autoregression (Spirals)')
plt.show()

#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
df = read_data_file(file_list[9])
radius, theta = unravel_spiral(df)
df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
df_polar_coordinate['Angle'] = theta
df_polar_coordinate['Radius'] = radius
Healthy_unravel = df_polar_coordinate['Radius'].tolist()
x= (np.diff(np.sign(Healthy_unravel)) != 0).sum()

# %%
