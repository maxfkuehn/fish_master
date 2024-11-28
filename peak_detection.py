import os
import numpy as np
from IPython import embed
from scipy.signal import find_peaks

from tqdm import tqdm
import matplotlib.pyplot as plt
from random import random
import sys
import matplotlib.dates as mdates
import datetime as dt
from fish_list_unpacker import fish_list_unpacker as flu
from thunderfish.eventdetection import detect_peaks 

########## loading data ############
#load fishlist
#auswahl datensatz a=0  a=1
a=0

#laodpath
load_path = '/home/kuehn/Max_Masterarbeit/data/processed_raw_data'

#### date time ######
# definiere Zeitpunkte der Datei als Datum (datetime):
start_date_0 = dt.datetime(2019, 10, 21, 13, 25, 00)
start_date_1 = dt.datetime(2019, 10, 22, 8, 13, 00)
record_day_0 = '/2019-10-21-13_25'
record_day_1 = '/2019-10-22-8_13'

if a == 0:
    start_date = start_date_0
    record_day = record_day_0

elif a == 1:
    start_date = start_date_1
    record_day = record_day_1


# load data files
if a == 0:
#21.10.19
    file = sorted(os.listdir(load_path))[0]
    fish_list = np.load(load_path+'/'+file+'/fishlist.npy', allow_pickle=True)

if a == 1:
    #22.10.19
    file= sorted(os.listdir(load_path))[1]
    fish_list= np.load(load_path+'/'+file+'/fishlist.npy', allow_pickle=True)

# laod full data out of fishlist via  fish list unpacker 

filepath= load_path+'/'+file
dic = flu(fish_list, filepath)

ident_af = dic['ident_list']
freq_af = dic['frequence_list']
time_af = dic['time_list']
time_idx_af = dic['time_idx_list']
ch_af = dic['channel_list']
sign_af =dic['sign_list']
times = dic['record_time_array']


# get freq and time for individual fish and detect peaks
 
for t in range(len(time_af)):
    freq = freq_af[t]
    time = time_af[t]
    # calculate day time array
   
    
    # sort data regarding time 
    nested_data = list(zip(time,freq))
    sorted_data = sorted(nested_data)

    time,freq = zip(*sorted_data)
    dt_time = [start_date + dt.timedelta(seconds=v) for v in time]
    # scipy.signals
    peaks = find_peaks(freq, prominence=5)

    prominence=peaks[1]['prominences']

    time_peaks = []
    freq_peaks = []

    for i in peaks[0]:
        time_peaks.append(dt_time[i])
        freq_peaks.append(freq[i])

    # thunderfish_peaks
    tf_peaks = detect_peaks(freq,5)[0]
 
   
    tf_time = []
    tf_freq = []
    
    for o in tf_peaks : 
        tf_time.append(dt_time[o])
        tf_freq.append(freq[o])

    
    # compare and plot peak detection
    peaks_fig, (ax1, ax2) = plt.subplots(1,2,sharex=True, sharey=True)

    date_format = mdates.DateFormatter('%H:%M:%S')

    ax1.plot(dt_time,freq,color='blue')
    ax1.plot(time_peaks,freq_peaks,color='red',marker='*',markersize=10, linestyle='')
    ax1.set_title('Scipy')
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(date_format)

    ax2.plot(dt_time,freq,color='blue')
    ax2.plot(tf_time,tf_freq,color='red',marker='*',markersize=10, linestyle='')
    ax2.set_title('thunderfish')
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(date_format)

plt.show()




