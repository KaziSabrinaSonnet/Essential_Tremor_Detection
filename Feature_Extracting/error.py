#%%
import pandas as pd
import numpy as np
import random
import os
import seaborn as sns
from matplotlib import pyplot as plt
import math
from scipy import signal
from scipy.signal import find_peaks
from random import sample
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
"""
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
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData3')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        df_X_ideal= df['XStart']
        df_Y_ideal= df['YEnd']

#%%
def find_t_values():
    df_Y_t= df_Y_tremor.tolist()
    df_X_t= df_X_tremor.tolist()
    X1T= []
    X2T= []
    for item in df_Y_t:
        if item<140:
            X1T.append(item)
    for item in df_Y_t: 
        if 140<item<190:
            X2T.append(item)
    return X1T, X2T
#%%
def find_h_values():
    df_Y_h= df_Y_healthy.tolist()
    df_X_h= df_X_healthy.tolist()
    X1H= []
    X2H= []
    for item in df_Y_h:
        if item<140:
            X1H.append(item)
    for item in df_Y_h:
        if 140<item<190:
            X2H.append(item)
    return X1H, X2H
#%%

ideal_tremor_X = list((a_i - b_i)**2 for a_i, b_i in zip(df_X_ideal, df_X_tremor))
ideal_tremor_Y = list((a_i - b_i)**2 for a_i, b_i in zip(df_Y_ideal, df_Y_tremor))
ideal_tremor = list(a_i + b_i for a_i, b_i in zip(ideal_tremor_X, ideal_tremor_Y))
it = list(int(math.sqrt(a)) for a in ideal_tremor)
#%%
ideal_healthy_X = list((a_i - b_i)**2 for a_i, b_i in zip(df_X_ideal, df_X_healthy))
ideal_healthy_Y = list((a_i - b_i)**2 for a_i, b_i in zip(df_Y_ideal, df_Y_healthy))
ideal_healthy = list(a_i + b_i for a_i, b_i in zip(ideal_healthy_X, ideal_healthy_Y))
ih = list(int(math.sqrt(a)) for a in ideal_healthy)

#%%
def masks1(vec):
    d = np.diff(vec)
    dd = np.diff(d)

    # Mask of locations where graph goes to vertical or horizontal, depending on vec
    to_mask = ((d[:-1] != 0) & (d[:-1] == -dd))
    # Mask of locations where graph comes from vertical or horizontal, depending on vec
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask

#%%
def masks2(vec):
    d = np.diff(vec)
    dd = np.diff(d)

    # Mask of locations where graph goes to vertical or horizontal, depending on vec
    to_mask = ((d[:-1] != 0) & (d[:-1] == -dd))
    # Mask of locations where graph comes from vertical or horizontal, depending on vec
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask
#%%
from matplotlib import pylab
from matplotlib.font_manager import FontProperties
def apply_mask2(mask, x, y):
    return x[1:-1][mask], y[1:-1][mask]
def apply_mask_healthy():
    to_vert_mask2, from_vert_mask2 = masks2(df_X_healthy)
    to_horiz_mask2, from_horiz_mask2 = masks2(df_Y_healthy)
    list1h, list2h = find_h_values()
    to_vert_t2, to_vert_v2 = apply_mask2(to_vert_mask2, df_X_healthy, df_Y_healthy)
    from_vert_t2, from_vert_v2 = apply_mask2(from_vert_mask2, df_X_healthy, df_Y_healthy )
    to_horiz_t2, to_horiz_v2 = apply_mask2(to_horiz_mask2, df_X_healthy, df_Y_healthy)
    from_horiz_t2, from_horiz_v2 = apply_mask2(from_horiz_mask2, df_X_healthy, df_Y_healthy)
    pylab.subplot(2,1,1)
    pylab.plot(df_X_healthy[0:len(list1h)], df_Y_healthy[0:len(list1h)], 'b-')
    pylab.plot(df_X_healthy[len(list1h):len(list1h)+len(list2h)], df_Y_healthy[len(list1h):len(list1h)+len(list2h)], 'b-')
    pylab.plot(df_X_healthy[len(list1h)+len(list2h):], df_Y_healthy[len(list1h)+len(list2h):], 'b-')
    #plt.plot(df_X_healthy, df_Y_healthy, 'b-')
    #plt.plot(to_vert_t2, to_vert_v2, 'r^', label='Plot goes vertical')
    #plt.plot(from_vert_t2, from_vert_v2, 'kv', label='Plot stops being vertical')
    #plt.plot(to_horiz_t2, to_horiz_v2, 'g>', label='Plot goes horizontal')
    pylab.plot(from_horiz_t2, from_horiz_v2, 'r<', label='Plot stops being horizontal')
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.3), ncol=2)
    pylab.title("Healthy")
    pylab.show()
#%%
from matplotlib import pylab
from matplotlib.font_manager import FontProperties
def apply_mask1(mask, x, y):
    return x[1:-1][mask], y[1:-1][mask]
def apply_mask_tremor():
    list1t, list2t = find_t_values()
    to_vert_mask1, from_vert_mask1 = masks1(df_X_tremor)
    to_horiz_mask1, from_horiz_mask1 = masks1(df_Y_tremor)
    to_vert_t1, to_vert_v1 = apply_mask1(to_vert_mask1, df_X_tremor, df_Y_tremor)
    from_vert_t1, from_vert_v1 = apply_mask1(from_vert_mask1, df_X_tremor, df_Y_tremor )
    to_horiz_t1, to_horiz_v1 = apply_mask1(to_horiz_mask1, df_X_tremor, df_Y_tremor)
    from_horiz_t1, from_horiz_v1 = apply_mask1(from_horiz_mask1, df_X_tremor, df_Y_tremor)
    pylab.subplot(2,1,1)
    pylab.plot(df_X_tremor[0:len(list1t)], df_Y_tremor[0:len(list1t)], 'b-')
    pylab.plot(df_X_tremor[len(list1t):len(list1t)+len(list2t)], df_Y_tremor[len(list1t):len(list1t)+len(list2t)], 'b-')
    pylab.plot(df_X_tremor[len(list1t)+len(list2t):], df_Y_tremor[len(list1t)+len(list2t):], 'b-')
    #plt.plot(to_vert_t1, to_vert_v1, 'r^', label='Plot goes vertical')
    # #plt.plot(from_vert_t1, from_vert_v1, 'kv', label='Plot stops being vertical')
    # #plt.plot(to_horiz_t1, to_horiz_v1, 'g>', label='Plot goes horizontal')
    pylab.plot(from_horiz_t1, from_horiz_v1, 'r<', label='Plot stops being horizontal')
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.3), ncol=2)
    pylab.title("DBS Stim Off")
    pylab.show()
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        df_X_tremor= df['XStart'].to_numpy()
        df_Y_tremor= df['YEnd'].to_numpy()
        apply_mask_healthy()
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
df = read_data_file(file_list[15])
df_X_tremor= df['XStart'].to_numpy()
df_Y_tremor= df['YEnd'].to_numpy()
apply_mask_healthy()
"""
df_X_tremor=(sample(df_X_tremor,144)) 
df_Y_tremor= (sample(df_Y_tremor,144)) 
"""
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
df = read_data_file(file_list[10])
df_X_healthy= df['XStart'].to_numpy()
df_Y_healthy= df['YEnd'].to_numpy() 
apply_mask_healthy()
"""
df_X_healthy=(sample(df_X_healthy,144)) 
df_Y_healthy= (sample(df_Y_healthy,144)) 
"""
#%%
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
df = read_data_file(file_list[9])
df_X_healthy= df['XStart'].tolist()
df_Y_healthy= df['YEnd'].tolist()
#%%

        
# %%
