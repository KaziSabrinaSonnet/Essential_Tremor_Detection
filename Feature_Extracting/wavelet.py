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
#%%
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
def read_data_file2(path_file):
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
import cmath 
def forming_signal(df):
    Signal= []
    Timestamps = []
    Theta = []
    for i in range(0, len(df)):
        Signal.append(math.sqrt(df['XStart'][i]**2 + df['YStart'][i]**2))
        Timestamps.append(df['TimeStamp'][i])
    return Signal, Timestamps
#%%
def calculate_sampling_frequency(path_file):
    df = read_data_file(path_file)
    number_of_samples= len(df)
    df_list2= df['TimeStamp'].tolist()
    total_time= (df_list2[len(df_list2)-1]-df_list2[0]) 
    sampling_frequency= number_of_samples/total_time
    return sampling_frequency
#%%
import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.fftpack import fft
from IPython.display import display
import scipy.stats
import datetime as dt
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
#%%
from scipy import stats
def bayesian_confidence_interval(list_values):
    res_mean, res_var, res_std = scipy.stats.bayes_mvs(list_values, alpha=0.95)
    return res_mean.statistic, res_mean.minmax[0],res_mean.minmax[1], res_std.statistic, res_std.minmax[0], res_std.minmax[1], res_var.statistic, res_var.minmax[0], res_var.minmax[1]


def calculate_statistics(list_values):
    coefficient_of_Variation = scipy.stats.variation(list_values)
    inter_quartile_range = scipy.stats.iqr(list_values)
    kstat= scipy.stats.kstat(list_values)
    standard_error_of_mean= stats.sem(list_values)
    median_absolute_deviation= stats.median_absolute_deviation(list_values)
    return coefficient_of_Variation, inter_quartile_range, kstat, standard_error_of_mean, median_absolute_deviation


def get_features(list_values):
    bci = list(bayesian_confidence_interval(list_values))
    statistics = list(calculate_statistics(list_values))
    return bci+statistics

#%%
import pywt
base_folder = "D:\\Masters@UW\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
Final_Features = []
Labels = []
#%%
##list
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
df = read_data_file(file_list[20])
signal, timestamps = forming_signal(df)
with open('Healthy20.txt', 'w') as filehandle:
    for listitem in signal:
        filehandle.write('%s\n' % listitem)
#%%
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df1 = read_data_file(file)
        P1= np.mean(df1['Pressure'].tolist())
        signal1, timestamps1= forming_signal(df1)
        list_coeff1 = pywt.wavedec(signal1, 'db1', level=5)
        list_coeff1 = list(list_coeff1)
        Features= get_features(list_coeff1[0])+get_features(list_coeff1[1])+get_features(list_coeff1[2])+get_features(list_coeff1[3])+P1
        Final_Features.append(Features)
        Labels.append('Healthy')
#%%
data_path=os.path.join(base_folder, 'StimOff')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
       df1 = read_data_file(file)
       P1= np.mean(df1['Pressure'].tolist())
       signal1, timestamps1= forming_signal(df1)
       list_coeff1 = pywt.wavedec(signal1, 'db1', level=5)
       list_coeff1 = list(list_coeff1)
       Features= get_features(list_coeff1[0])+get_features(list_coeff1[1])+get_features(list_coeff1[2])+get_features(list_coeff1[3])+P1
       Final_Features.append(Features)
       Labels.append('Untreated')
#%%
data_path=os.path.join(base_folder, 'StimOn')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df1 = read_data_file(file)
        P1= np.mean(df1['Pressure'].tolist())
        signal1, timestamps1= forming_signal(df1)
        list_coeff1 = pywt.wavedec(signal1, 'db1', level=5)
        list_coeff1 = list(list_coeff1)
        Features= get_features(list_coeff1[0])+get_features(list_coeff1[1])+get_features(list_coeff1[2])+get_features(list_coeff1[3])+P1
        Final_Features.append(Features)
        Labels.append('Treated')
#%%
def get_train_test(df, y_col, x_cols, ratio):
    mask = np.random.rand(len(df))<ratio
    df_test = df[mask]
    df_train = df[~mask]
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test
#%%
df = pd.DataFrame(Final_Features)
ycol = 'y'
xcols = list(range(df.shape[1]))
df.loc[:,ycol] = Labels
df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, ycol, xcols, ratio = 0.3)

# %%
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("The Train Score is {}".format(train_score))
print("The Test Score is {}".format(test_score))
#%%
from scipy import stats
import pywt
from scipy.stats import moment
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
Final_Features = []
Labels = []
healthy_a = []
data_path=os.path.join(base_folder, 'TabletData2')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        signal, time = forming_signal(df)
        """
        w = pywt.Wavelet('db1')
        m= pywt.dwt_max_level(data_len=len(signal), filter_len=w.dec_len)
        print(m)
        """
        a = stats.median_absolute_deviation(signal)
        healthy_a.append(a)
#%%
import pywt
base_folder = "D:\\Autumn 2019\\Research\\Feature_Extracting"
COLUMN_NAMES = ['TimeStamp', 'XEnd', 'YEnd', 'XStart', 'YStart', 'Pressure']
Final_Features = []
Labels = []
tremor_a = []
data_path=os.path.join(base_folder, 'TabletData1')
file_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
for file in file_list:
    if os.path.isfile(file):
        df = read_data_file(file)
        signal, time = forming_signal(df)
        """
        w = pywt.Wavelet('db1')
        m= pywt.dwt_max_level(data_len=len(signal), filter_len=w.dec_len)
        print(m)
        """
        a = stats.median_absolute_deviation(signal)
        tremor_a.append(a)
#%%
import matplotlib.pyplot as plt
plt.plot(healthy_a,'g*', tremor_a, 'ro')
plt.xticks([])
plt.title('')
plt.show()
#%%
import itertools
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
matrix = confusion_matrix(Y_test,y_pred)
class_names = ['Healthy', 'StimOff', 'StimOn']
plt.clf()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
fmt = 'd'
thresh = matrix.max() / 2.
for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
    plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label',size=14)
plt.xlabel('Predicted label',size=14)
plt.show()
#%%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from matplotlib.patches import Circle

Feature1H = []
for item in Final_Features[0:45]:
    Feature1H.append(item[12])
Feature1T = []
for item in Final_Features[45:68]:
    Feature1T.append(item[12])
Feature1UT = []
for item in Final_Features[68:]:
    Feature1UT.append(item[12])
import matplotlib.pyplot as plt
"""
plt.plot(Feature1H, 'g*', Feature1UT, 'ro', Feature1T, 'b*')
plt.xticks([])
plt.title('Fractral Dimension')
plt.show()
"""
plt.figure(figsize=(4,6))
plt.plot(np.zeros(np.shape(Feature1H)),Feature1H, 'g*')
plt.plot(np.zeros(np.shape(Feature1T)), Feature1T, 'b*')
plt.plot(np.zeros(np.shape(Feature1UT)), Feature1UT, 'ro')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
legend_elements = [Line2D([0], [0], marker='*', color='w', label='Healthy',
                          markerfacecolor='g', markersize=10),
                   Line2D([0], [0], marker='*', color='w', label='Treated',
                          markerfacecolor='b', markersize=10), Line2D([0], [0], marker='o', color='w', label='Untreated',
                          markerfacecolor='r', markersize=5), ]

plt.legend(handles=legend_elements, loc='upper left')
plt.show()
#%%
y = []
for i in range(len(Labels)):
    if Labels[i] == 'Healthy':
        y.append(0)
    elif Labels[i] == 'StimOff':
        y.append(1)
    elif Labels[i] == 'StimOn':
        y.append(2)
X= Final_Features
X= np.array(X) 
y= np.array(y) 
target_names= np.array(['Healthy', 'StimOff', 'StimOn'])
#%%
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_r2 = lda.fit_transform(X, y)
plt.figure()
colors = ['green', 'red', 'blue']
point1 = [-4, -4]
point2 = [5, 3]
point3= [1, -0.1]
point4= [-2, 5]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
x1_values = [point3[0], point4[0]]
y1_values = [point3[1], point4[1]]
plt.plot(x_values, y_values)
plt.plot(x1_values, y1_values)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Linear Discriminant Analysis')
"""
plt.xticks([])
plt.yticks([])
"""
plt.xlabel("LDA component 1")
plt.ylabel("LDA component 2")
plt.show([])
# %%



# %%
