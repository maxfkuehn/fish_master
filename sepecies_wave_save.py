import datetime as dt
import math
import os
import random as rn
import statistics as stc
from thunderlab.dataloader import DataLoader
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import audioio as aio
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from IPython import embed
from tqdm import tqdm

# load own functions
from fish_list_unpacker import fish_list_unpacker as flu



def find_consecutive_diff_sequence( freq, sign, time, ch,fish):

    #returns the index of the first 5 consecutive elments of an array
    consecutive_sequence = []
    threshhold = np.arange(-100,-20)[::-1]
    start_index = []
    ban_list=[]
    lasttime = None

    zip_data = list(zip(freq,sign,time,ch))
    sorted_zip = sorted(zip_data, key=lambda x: x[0])

    ch_sorted_time = [item[0] for item in sorted_zip]
    ch_sorted_freq = [item[1] for item in sorted_zip]
    ch_sorted_ch = [item[2] for item in sorted_zip]
    ch_sorted_sign = [item[3] for item in sorted_zip]    


    for th in threshhold:
        for i in range(len(ch_sorted_ch)-1):
            if ch_sorted_ch[i+1] == ch[i]:
                if np.diff([ch_sorted_time[i], ch_sorted_time[i+1]]) < 0.3 and ch_sorted_sign[i+1] > th:
                    if lasttime is not None and time[i] - lasttime < 10:
                        continue
                    start_index.append(i)
                    if len(start_index) == 3:
                        return start_index
    return start_index

  

def NooverlappInTimeMaxAmplitudeFinder(freq,sign, time,ch):
# finds up to the 3 biggest amplitudes with atleast X seconds time difference if possible
    # input freq list, sign list, time list and ch list of one fish
    # output time_point_list, ch_list and freq_list
    # time_points_list: time pooints of highest amplitude with atleast 20s apart from each other
    #ch_list: channel of data points
    #frq_list: freq at data time points

    # sorts sign and reorders the other values according to the corresponding element
    zip_data = list(zip(freq,sign,time,ch))
    sorted_zip = sorted(zip_data, key=lambda x: x[1], reverse=True)

    s_sorted_time = [item[2] for item in sorted_zip]
    s_sorted_freq = [item[0] for item in sorted_zip]
    s_sorted_ch = [item[3] for item in sorted_zip]
    s_sorted_sign = [item[1] for item in sorted_zip]    
    
    
    t_diff_min = 20 # sets the minimal difference in time [s] between two viable data points
    time_idx_list = []
    ch_list_idx = []
    frq_list =[]
    
    # goes throught list of sign from highest to lowest amplitude 
    #saves the  time of the first highest amplitude data point with corresponding ch and freq
    # then compares the other amplitudes and checks if ch is different or time difference is greater then 20 compared to previous saved data points
    # if one is true time, chh and freq are also saved

    for s in range(len(s_sorted_sign)): 

        current_ch = s_sorted_ch[s]
        test_time = s_sorted_time[s]
        same_channel = any(t == current_ch for t in ch_list_idx)
        current_frq = s_sorted_freq[s]

        if len(time_idx_list) == 3:
            break    


        elif s == 0:
            time_idx_list.append(test_time)
            ch_list_idx.append(current_ch)
            frq_list.append(current_frq)
        elif same_channel == True:
            
            ch_time_idx = np.where(np.array(ch_list_idx)== current_ch)[0]
            ch_time = np.array(time_idx_list)[ch_time_idx]
            
            time_diff = all(abs(tt-test_time)>= t_diff_min for tt in ch_time)

            if time_diff == True:
                time_idx_list.append(test_time)
                ch_list_idx.append(current_ch)
                frq_list.append(current_frq)  
            else:
                continue
        else:
            time_idx_list.append(test_time)
            ch_list_idx.append(current_ch)
            frq_list.append(current_frq)            
    
    return time_idx_list, ch_list_idx, frq_list

def SpeciesIdentificationWavCreator(master,slave, freq_af, time_af, ch_af, sign_af,day):
    
    for fish in range(len(freq_af)):
        # sort time array and reorder other lists according to the corresponding to time value of each element
        zip_data = list(zip(time_af[fish], freq_af[fish], ch_af[fish], sign_af[fish]))
        sorted_zip = sorted(zip_data, key=lambda x: x[0])

        sorted_time = [item[0] for item in sorted_zip]
        sorted_freq = [item[1] for item in sorted_zip]
        sorted_ch = [item[2] for item in sorted_zip]
        sorted_sign = [item[3] for item in sorted_zip]

        time, chanl, frq = np.asarray(NooverlappInTimeMaxAmplitudeFinder(sorted_freq ,sorted_sign ,sorted_time, sorted_ch,fish))
        
        counter = 0

        if time.size >1:
            for fi in range(len(time)):

                counter += 1
                sel_ch = chanl[fi]

                time_first = (time[fi]-10)*30000

                if time_first < 0:
                    time_first = 0

                time_end = (time[fi]+10)*30000
                embed()
                if sel_ch < 15:
                    amp_raw = master[time_first:time_end]
                    ch = sel_ch
                else:
                    amp_raw = slave[time_first:time_end]
                    ch = sel_ch-16

                freq = frq[fi]
                data = [amp[int(ch)] for amp in amp_raw]

                aio.write_audio(f'/home/kuehn/data/waveforms/waveform_day{day}_{fish:03d}_{counter}_{freq:.0f}.wav', data, 30000)
    print(fish)






if __name__ == "__main__":
    
    #Aim of this cript is to create a wav file to be used in Thunderfish by Jan Benda to get EODf waveforsm.
    #This scripts looks for the four not overlapping timewindows in which the recorded amplitude of the fish was the highest 
    #but also robust and lasted the whole time window. Those results are then saved as wav data at 
    
    # auswahl datensatz a=0 21.10.19 a=1 22.10.19
    auswahl = [0]  # select record day

    # min amplitude to count t [in mV]
    Vmin = 1  # [microV]
    # amount of highest amplitude counted
    X = 4
    # save path
    # load fishlist
    #### date time ######
    # definiere Zeitpunkte der Datei als Datum (datetime):
    start_date_0 = dt.datetime(2019, 10, 21, 13, 25, 00)
    start_date_1 = dt.datetime(2019, 10, 22, 8, 13, 00)
    record_day_0 = '/2019-10-21-13_25'
    record_day_1 = '/2019-10-22-08_13'

    # list of species of each fish over both days
    species_af = []

    for a in auswahl:
        if a == 0:
            start_date = start_date_0
            record_day = record_day_0
        elif a == 1:
            start_date = start_date_1
            record_day = record_day_1

        ##### import fish data:#######

        # save path
        load_path = '/home/kuehn/data/processed_raw_data'
    
        filename = sorted(os.listdir(load_path))[a]
        fish_list = np.load(load_path + '/' + filename + '/fishlist.npy', allow_pickle=True)
        filepath = load_path + '/' + filename
        save_date = record_day.replace('/', '')

        # load data
        dic = flu(fish_list, filepath)

        ident_af = dic['ident_list']
        freq_af = dic['frequence_list']
        time_af = dic['time_list']
        time_idx_af = dic['time_idx_list']
        ch_af = dic['channel_list']
        sign_af = dic['sign_list']
        times = dic['record_time_array']
        sign_af_miV = dic['sign_list_microV']

        load_master = '/home/kuehn/data/mount_data/master' + record_day + '/traces-grid1.raw'

        load_slave = '/home/kuehn/data/mount_data/slave' + record_day +'/traces-grid1.raw'


        #raw_data_master = DataLoader(load_master, 60., 10., -1)
        #raw_data_slave = DataLoader(load_slave, 60., 10., -1)
        raw_data_master = []
        raw_data_slave  = []
        species_list = SpeciesIdentificationWavCreator(raw_data_master, raw_data_slave, freq_af, time_af, ch_af, sign_af,a)

      
    