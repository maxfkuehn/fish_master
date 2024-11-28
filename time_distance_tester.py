import datetime as dt
import math
import os
import random as rn
import statistics as stc

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from IPython import embed
from tqdm import tqdm

# load own functions
from fish_list_unpacker import fish_list_unpacker as flu

# Chekcs single data points if there are to big time gaps towards other points, which catogorices them as noice
# Also Cheks if when a fish reaappeared with same freq if the position it appeared made sense or its another fish

if __name__ == "__main__":
    # auswahl datensatz a=0 21.10.19 a=1 22.10.19
    a = 0  # select record day

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
    record_day_1 = '/2019-10-22-8_13'

    if a == 0:
        start_date = start_date_0
        record_day = record_day_0
    elif a == 1:
        start_date = start_date_1
        record_day = record_day_1

    ##### import fish data:#######

    # save path
    load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'

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

    # get max ident
    max_ident = []
    for ident in ident_af:
        max_ident.append(max(ident))
    max_ident = max(max_ident)

    # check for temporal singled out data points
    fish_nr = -1
    x = 5
    delete_t_points = []
    for time_list in time_af:  # loop over every fish and get their time list
        fish_nr += 1
        time_points = sorted(set(time_list))  # get list of unique time points
        solo_point
        for t in range(1,len(time_points[:-1])): #not checking first and last point in loop
            diff_before = time_points[t]-time_points[t-1]
            diff_after = time_points[t+1]-time_points[t]

            if diff_before >= x and diff_after >= x:
                solo_point.append(time_points[t])
