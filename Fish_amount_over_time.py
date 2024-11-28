import os
import math
import statistics as stc
import datetime as dt
import random as rn

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
from IPython import embed
from tqdm import tqdm
# load own functions
from fish_list_unpacker import fish_list_unpacker as flu



def fish_amount_over_time(fish_list,bin_width,filepath):
    #unpack fish_list
    dic = flu(fish_list, filepath)

    ident_af = dic['ident_list']
    freq_af = dic['frequence_list']
    time_af = dic['time_list']
    time_idx_af = dic['time_idx_list']
    ch_af = dic['channel_list']
    sign_af =dic['sign_list']
    times = dic['record_time_array']


    #histogramm variables
    1

    #check fpr every bin if individual fish appears
    start_bin = 0
    end_bin = bin_width
    # array with presence of a bin in fish
    hist_array = []
    # binary presence check for every bin for every fish
    presence_af = []

    # Iterate over every fish
    for fish_nr in tqdm(range(len(time_af))):
        sf_presence = [0] * bin_number  # Initialize presence list for each fish
        # Loop for every bin
        for bin in range(bin_number):
            # Check if any time within the bin range
            presence = any((start_bin + bin * bin_width <= t <= end_bin + bin * bin_width) for t in time_af[fish_nr])
            if presence:
                hist_array.append(bin)
                sf_presence[bin] = 1

        presence_af.append(sf_presence)

    np.save((filepath+'/histogram_array_' + str(bin_width) +'.npy'),hist_array)
    np.save((filepath + '//bin_number_' + str(bin_width) + '.npy'), bin_number)
    np.save((filepath + '/presence_all_ fish' + str(bin_width) +'.npy'), presence_af)
    return hist_array , bin_number

def histogram_maker(histogram_array, bin_width, bin_number, start_date):
    # Create record date for title
    record_day = start_date.strftime('%d.%m.%y %H:%M')
    # set sunrise and sunset time as dt
    sunrise_time = dt.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=5, minute=34)
    sunset_time = dt.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=17, minute=54)
    num_sunset_times = range(-1,2)
    sum_sunrise_times = range(2)
    #calculate multipleonsets
    sunset_times = []
    rise_times =[]

    for offset in num_sunset_times:

        new_sunset_time = sunset_time + dt.timedelta(days=offset)
        sunset_times.append(new_sunset_time)

    for rise_offset in sum_sunrise_times:
        new_sunrise_time = sunrise_time + dt.timedelta(days=rise_offset)
        rise_times.append(new_sunrise_time)


    # Generate an array of datetime values for the bins
    bins_dt = [start_date + dt.timedelta(seconds=bin_width * i) for i in range(bin_number)]
    bins_num = mdates.date2num(bins_dt)  # Convert datetime objects to numerical representation
    #generate countbin for histogrammaker
    counts, _ = np.histogram(histogram_array, bins = bin_number)
    # Plot the histogram
    fig, ax = plt.subplots()
    ax.hist(bins_num, bins=bin_number, weights=counts)

    ax.set_title(f'Number of fishes over time in {bin_width}s bins\nRecord start: {record_day}')
    ax.set_ylabel('Number of fish present [#]')
    ax.set_xlabel(f'Time in {bin_width}s bins')

    # Format the x-axis labels to display only hours and minutes
    date_format = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    # Format Xlim
    ax.set_xlim(bins_num[0], bins_num[-1])
    #ax.axvspan(sunset_times[0], rise_times[0], facecolor = 'black',alpha=0.2)
    ax.axvspan(sunset_times[1], rise_times[1], facecolor='black', alpha=0.2)
    #savefig and show
    fig.savefig(f"Fish_#_over_{bin_width}s_{start_date}.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    ########## settings ############



    # auswahl datensatz a=0 21.10.19 a=1 22.10.19
    a = 1  # select record day
    # data hist ready
    b = 0 # 0 no, 1 yes
    # save path
    load_path = '/home/kuehn/Max_Masterarbeit/data/processed_raw_data' 
    # load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data' 
    # histogram settings
    bin_width = 10 #in seconds

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
    #### importiert electrode position grid ####
    #from recording_grid_columbia_2019 import x_grid
    #from recording_grid_columbia_2019 import y_grid
    #from recording_grid_columbia_2019 import electrode_id_list
    ##### import fish data:#######


    filename = sorted(os.listdir(load_path))[a]
    fish_list = np.load(load_path +'/'+ filename +'/fishlist.npy', allow_pickle=True)
    filepath = load_path +'/'+ filename
    if b == 0:
        hist_array, bin_number = fish_amount_over_time(fish_list, bin_width, filepath)
        histogram_maker(hist_array,bin_width,bin_number, start_date)
    else:
        hist_load = '/histogram_array_'+str(bin_width) + '.npy'
        bin_load = '/bin_number_'+str(bin_width) + '.npy'
        hist_array = np.load(load_path + '/' + filename + hist_load, allow_pickle=True)
        bin_number = np.load(load_path + '/' + filename + bin_load, allow_pickle=True)
        histogram_maker(hist_array,bin_width,bin_number, start_date)