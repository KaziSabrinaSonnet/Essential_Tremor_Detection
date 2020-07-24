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
    lookahead = 25
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
#%%
def plot_unravelled_spiral(df, path_file):
        fig = sns.lineplot(df['Angle'], df['Radius'])
        fig.set(xlabel='Angle', ylabel='Radius')
        path_folder = os.path.join(data_path, "plots_unravelled_spirals")
        create_folder(path_folder)
        fig.get_figure().savefig(
            os.path.join(path_folder,
            os.path.basename(path_file) + '_unravelling.png'))
        plt.show()
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
mean_list_tremor = []
for file in file_list:
    if os.path.isfile(file):
        tremor_list= []
        i=0
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        df_polar_coordinate['Radius'] = radius
        Tremor_theta= df_polar_coordinate['Angle'].tolist()
        Tremor_radius= df_polar_coordinate['Radius'].tolist()
        for i in range(0, len(Tremor_theta)-1):
            if (Tremor_theta[i+1]-Tremor_theta[i])!=0:
                tremor_list.append((abs(Tremor_radius[i+1]-Tremor_radius[i])/(Tremor_theta[i+1]-Tremor_theta[i])))
                i= i+1
        mean_list_tremor.append(np.mean(tremor_list))
        
        plt.xlabel ('Points')
        plt.ylabel('|dr/dθ|')
        plt.ylim (0, 100000)
        #plt.plot(sample(tremor_list, 50),'r')
        plt.plot(tremor_list, 'r')
        plt.show()  
        
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
mean_list_healthy = []
for file in file_list:
    if os.path.isfile(file):
        healthy_list= []
        i=0
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        df_polar_coordinate['Radius'] = radius
        Healthy_theta= df_polar_coordinate['Angle'].tolist()
        Healthy_radius= df_polar_coordinate['Radius'].tolist()
        for i in range(0, len(Healthy_theta)-1):
            if (Healthy_theta[i+1]-Healthy_theta[i])!=0:
                healthy_list.append((abs(Healthy_radius[i+1]-Healthy_radius[i])/(Healthy_theta[i+1]-Healthy_theta[i])))
                i= i+1
        mean_list_healthy.append(np.mean(healthy_list))
        
        plt.xlabel ('Points')
        plt.ylabel('|dr/dθ|')
        plt.ylim (0, 100000)
        #plt.plot(sample(tremor_list, 50),'r')
        plt.plot(healthy_list, 'g')
        plt.show()
        

#%%
mean_list_healthy.remove(-16518.95984240397)
mean_list_tremor.remove(-12195.73925875211)
import matplotlib.pyplot as plt
plt.plot(mean_list_healthy,'g*', mean_list_tremor, 'ro')
plt.title('...')
plt.show()

        

#%%
data=tremor_list
Fs= 133
win = 4* Fs
freqs, psd = signal.welch(data, Fs, nperseg=win)
fig = plt.figure(figsize=(8, 4))
plt.plot(freqs, psd, lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's Velocity Spectrum (Healthy)")
plt.xlim([0, freqs.max()])
plt.show() 
     
#%%
#Ideal Spiral Accumulation Angle 
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData3')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Accumulation_Angle_Ideal_List =[]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        ideal_angle= df_polar_coordinate['Angle'].tolist()
        df_polar_coordinate['Radius'] = radius
        accumulation_angle_ideal = 0
        i= 0
        for elements in range (0, len(ideal_angle)-1): 
            accumulation_angle_ideal = accumulation_angle_ideal + abs(ideal_angle[i+1]-ideal_angle[i])
            i = i+1
        Accumulation_Angle_Ideal_List.append(accumulation_angle_ideal)
#%%
#Tremor Spiral Accumulation Angle 
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Accumulation_Angle_Tremor_List= []
Error_Tremor_List = []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        tremor_angle= df_polar_coordinate['Angle'].tolist()
        df_polar_coordinate['Radius'] = radius
        accumulation_angle_tremor = 0
        i= 0
        for elements in range (0, len(tremor_angle)-1): 
            accumulation_angle_tremor = accumulation_angle_tremor + abs(tremor_angle[i+1]-tremor_angle[i])
            #error_tremor = abs(Accumulation_Angle_Ideal_List[0]-accumulation_angle_tremor)
            error_tremor = abs((accumulation_angle_tremor-2.5)*8/35-accumulation_angle_tremor)
            i = i+1
        Accumulation_Angle_Tremor_List.append(accumulation_angle_tremor)
        Error_Tremor_List.append(error_tremor)

#%%
#Healthy Spiral Accumulation Angle 
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
Accumulation_Angle_Healthy_List= []
Error_Healthy_List = []
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        radius, theta = unravel_spiral(df)
        df_polar_coordinate = pd.DataFrame(columns=['Angle', 'Radius'])
        df_polar_coordinate['Angle'] = theta
        healthy_angle= df_polar_coordinate['Angle'].tolist()
        df_polar_coordinate['Radius'] = radius
        accumulation_angle_healthy = 0
        i= 0
        for elements in range (0, len(healthy_angle)-1): 
            accumulation_angle_healthy = accumulation_angle_healthy + abs(healthy_angle[i+1]-healthy_angle[i])
            #error_healthy = abs(Accumulation_Angle_Ideal_List[0]-accumulation_angle_healthy)
            error_healthy = abs((accumulation_angle_healthy-2.5)*8/35-accumulation_angle_healthy)
            i = i+1
        Accumulation_Angle_Healthy_List.append(accumulation_angle_healthy)
        Error_Healthy_List.append(error_healthy)
#%%
detrend_error_healthy = signal.detrend(np.array(Error_Healthy_List))
detrend_error_tremor = signal.detrend(np.array(Error_Tremor_List))
deviation_healthy = [abs(i - j) for i, j in zip(Error_Healthy_List, detrend_error_healthy.tolist())]
deviation_tremor = [abs(i - j) for i, j in zip(Error_Tremor_List, detrend_error_tremor.tolist())]
slope_healthy = abs(deviation_healthy[0]-deviation_healthy[len(deviation_healthy)-1])/Accumulation_Angle_Healthy_List[len(Accumulation_Angle_Healthy_List)-1]
slope_tremor = abs(deviation_tremor[0]-deviation_tremor[len(deviation_tremor)-1])/Accumulation_Angle_Tremor_List[len(Accumulation_Angle_Tremor_List)-1]
final_deviation_healthy = slope_healthy * Accumulation_Angle_Healthy_List[len(Accumulation_Angle_Healthy_List)-1]
final_deviation_tremor = slope_tremor * Accumulation_Angle_Tremor_List[len(Accumulation_Angle_Tremor_List)-1]

#%%
import matplotlib.pyplot as plt 
import numpy as np
left = [0.1, 0.2] 
height = [slope_healthy, slope_tremor] 
tick_label = ['Healthy', 'Tremor'] 
plt.bar(left, height, tick_label = tick_label, width = 0.05, color = ['green', 'red']) 
plt.xlabel('Total Deviation in the Slope of Accumulated Angle') 
plt.show() 
#%%
import matplotlib.pyplot as plt 
import numpy as np
left = [0.1, 0.2] 
height = [final_deviation_healthy, final_deviation_tremor] 
tick_label = ['DeviationHealthy', 'DeviationTremor'] 
plt.bar(left, height, tick_label = tick_label, width = 0.05, color = ['green', 'red']) 
plt.xlabel('Parameters') 
plt.show()

#%%
import matplotlib.pyplot as plt
 

 
box_plot_data=[slope_healthy, slope_tremor]
box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Healthy', 'Tremor']
            )
 
colors = ['green', 'blue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
 
plt.show()

# %%
