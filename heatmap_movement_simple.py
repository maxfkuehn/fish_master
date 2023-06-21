import datetime as dt
import math
import os
import random as rn
import statistics as stc

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.cbook import get_sample_data
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from IPython import embed
from tqdm import tqdm

# load own functions
from fish_list_unpacker import fish_list_unpacker as flu
#### importiert electrode position grid ####
from recording_grid_columbia_2019 import x_grid
from recording_grid_columbia_2019 import y_grid
from recording_grid_columbia_2019 import electrode_id_list

# auswahl datensatz a=0 21.10.19 a=1 22.10.19, a=2 both days
a = 2  # select record day
# auswahl heat_matrix already calculated no = 0, yes=1
b = 1
# auswahl in c= 0 unter einander geplotete, c=1 in einem plot
c = 1

# save path
load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'

# histogram settings
bin_width = 10  # in seconds

# Create a custom colormap
cmap = mpl.colormaps['plasma']  # Get the colormap
cmap.set_under(color='black', alpha=0.9)  # Set color for value 0 to black

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
elif a == 2:
    # manipulate date of day one to be on day date of date 2 so they overlap in plot
    start_date_manipulated = [dt.datetime(2019, 10, 22, 13, 25, 00),start_date_1]
    start_date = [start_date_0,start_date_1]
    record_day = [record_day_0,record_day_1]
##### import fish data:#######

if a == 2:
    b = 1
    filename = []
    filepath = []
    save_date = []
    bin_number = []
    for idx in range(len(start_date)):
        filename.append(sorted(os.listdir(load_path))[idx])
        filepath.append(load_path + '/' + filename[idx])
        save_date.append(record_day[idx].replace('/', ''))

        fish_list = np.load(load_path + '/' + filename[idx] + '/fishlist.npy', allow_pickle=True)
        dic = flu(fish_list, filepath[idx])

        times = dic['record_time_array']

        bin_number.append(math.ceil(max(times) / bin_width))
        # bin variable

else:
    filename = sorted(os.listdir(load_path))[a]
    fish_list = np.load(load_path + '/' + filename + '/fishlist.npy', allow_pickle=True)
    filepath = load_path + '/' + filename
    save_date = record_day.replace('/', '')
    print(f'{save_date}, bin:{bin_width}')
    # arangement of elektrodes from bottomto top

    dic = flu(fish_list, filepath)

    ident_af = dic['ident_list']
    freq_af = dic['frequence_list']
    time_af = dic['time_list']
    time_idx_af = dic['time_idx_list']
    ch_af = dic['channel_list']
    sign_af = dic['sign_list']
    times = dic['record_time_array']
    # bin variable
    bin_number = math.ceil(max(times) / bin_width)
    start_bin = 0
    end_bin = bin_width

# electrode arrangement
electrode_arrangement_list = [25, 17, 26, 18, 27, 19, 28, 20, 29, 21, 30, 22, 31, 23, 32, 24, 9, 1, 10, 2, 11, 3, 12, 4,
                                  13, 5, 14, 6, 15, 7, 16, 8]
# CAlculate Data
if b == 0:
    if a != 2:
    # raw, time in 60s bins, column: all elektrodes from one and to another, numbers of fish that were
    # measured at elektrode in 60s, individual fish onlycounted once

        heat_map_matrix_single = np.zeros((len(electrode_arrangement_list), bin_number), dtype=int)
        heat_map_matrix_combined = np.zeros((len(electrode_arrangement_list) // 2, bin_number), dtype=int)

        for bn in tqdm(range(bin_number)):
            for fish_nr in range(len(time_af)):
                # check for presence of fish in time bin
                presence = (time_af[fish_nr] >= start_bin + bn * bin_width) & (
                        time_af[fish_nr] <= end_bin + bn * bin_width)
                if any(presence):
                    hits = np.where(presence)[0]
                    unique_hits = list(set(ch_af[fish_nr][hits] + 1))
                    for hit in unique_hits:
                        # check on which electrode fishappeared
                        electrode_idx = np.where(np.array(electrode_arrangement_list) == hit)[0]
                        heat_map_matrix_single[electrode_idx, bn] += 1
                        heat_map_matrix_combined[electrode_idx // 2, bn] += 1

        np.save((filepath + '/heatmap_matrix_single_' + str(bin_width) + '.npy'), heat_map_matrix_single)
        np.save((filepath + '/heatmap_matrix_combined_' + str(bin_width) + '.npy'), heat_map_matrix_combined)
    else:
        print('Not working for both days selected, pls select only one day!')
# Load data
elif b == 1:
    #both days
    if a == 2:
        heat_map_matrix_single_d1 = np.load(filepath[0] + '/heatmap_matrix_single_' + str(bin_width) + '.npy')
        heat_map_matrix_combined_d1 = np.load(filepath[0] + '/heatmap_matrix_combined_' + str(bin_width) + '.npy')

        heat_map_matrix_single_d2 = np.load(filepath[1] + '/heatmap_matrix_single_' + str(bin_width) + '.npy')
        heat_map_matrix_combined_d2 = np.load(filepath[1] + '/heatmap_matrix_combined_' + str(bin_width) + '.npy')
    #single day
    else:
        heat_map_matrix_single = np.load(filepath + '/heatmap_matrix_single_' + str(bin_width) + '.npy')
        heat_map_matrix_combined = np.load(filepath + '/heatmap_matrix_combined_' + str(bin_width) + '.npy')

# Plot Heatmap

#both days
if a == 2:
    #plot both in one plot
    if c == 1:
        # create datetime arary from start date in range of number of bins
        datetimes_d1 = [(start_date[0] + dt.timedelta(seconds=bin_width * i)) for i in
                        range(bin_number[0])]
        datetimes_d2 = [(start_date[1] + dt.timedelta(seconds=bin_width * i)) for i in
                        range(bin_number[1])]

        # create 2d matrices
        x1, y1 = np.meshgrid(datetimes_d1, np.arange(0, len(electrode_arrangement_list)))
        xc1, yc1 = np.meshgrid(datetimes_d1, np.arange(0, int(len(electrode_arrangement_list) / 2)))
        x2, y2 = np.meshgrid(datetimes_d2, np.arange(0, len(electrode_arrangement_list)))
        xc2, yc2 = np.meshgrid(datetimes_d2, np.arange(0, int(len(electrode_arrangement_list) / 2)))

        fig_single, ax1 = plt.subplots()
        c11 = ax1.pcolormesh(x1, y1, heat_map_matrix_single_d1, cmap=cmap, shading='auto', vmin=0.1)
        c12 = ax1.pcolormesh(x2, y2, heat_map_matrix_single_d2, cmap=cmap, shading='auto', vmin=0.1)

        colorbar1 = fig_single.colorbar(c11, ax=ax1, location='right')
        colorbar1.set_label('Amount of fish present')
        # set labels
        ax1.set_title(f'Heatmap of presences of fish on single electrodes\n date: {save_date}')
        ax1.set_xlabel(f'Time in bins [{bin_width}s]')
        fig_single.supylabel('Electrode position [#]')


        fig_comb, ax2 = plt.subplots()
        c21 = ax2.pcolormesh(xc1, yc1, heat_map_matrix_combined_d1, cmap=cmap, shading='auto', vmin=0.1)
        c22 = ax2.pcolormesh(xc2, yc2, heat_map_matrix_combined_d2, cmap=cmap, shading='auto', vmin=0.1)

        colorbar2 = fig_single.colorbar(c21, ax=ax2, location='right')
        colorbar2.set_label('Amount of fish present')
        # set labels
        ax2.set_title(f'Heatmap of presences of fish on single electrodes\n date: {save_date}')
        ax2.set_xlabel(f'Time in bins [{bin_width}s]')
        fig_comb.supylabel('Electrode position [#]')

        # Format the x-axis labels to display only hours and minutes
        date_format = mdates.DateFormatter('%H:%M')

        # Formatiere die x-Achse für den ersten Plot
        ax1.xaxis_date()
        ax2.xaxis_date()
        ax1.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_formatter(date_format)

        fig_comb.savefig(f"Heatmap_comb_{bin_width}s_both_dates_together.png", dpi=600)
        fig_single.savefig(f"Heatmap_single_{bin_width}s_both_dates_together.png", dpi=600)
        plt.show()

        plt.show()
    #plot both beneath each other
    else:
        datetimes_d1 = [(start_date_manipulated[0] + dt.timedelta(seconds=bin_width * i)) for i in
                        range(bin_number[0] )]
        datetimes_d2 = [(start_date_manipulated[1] + dt.timedelta(seconds=bin_width * i)) for i in
                        range(bin_number[1] )]

        # create 2d matrices
        x1, y1 = np.meshgrid(datetimes_d1, np.arange(0, len(electrode_arrangement_list)))
        xc1, yc1 = np.meshgrid(datetimes_d1, np.arange(0, int(len(electrode_arrangement_list) / 2)))
        x2, y2 = np.meshgrid(datetimes_d2, np.arange(0, len(electrode_arrangement_list)))
        xc2, yc2 = np.meshgrid(datetimes_d2, np.arange(0, int(len(electrode_arrangement_list) / 2)))

        # Plot the heatmaps with a normalized colormap
        fig_single, (ax11, ax12) = plt.subplots(2, 1, sharex=True)

        c11 = ax11.pcolormesh(x1, y1, heat_map_matrix_single_d1, cmap=cmap, shading='auto', vmin=0.1)
        c12 = ax12.pcolormesh(x2, y2, heat_map_matrix_single_d2, cmap=cmap, shading='auto', vmin=0.1)
        # place colorbar
        colorbar2 = fig_single.colorbar(c11, ax=[ax11, ax12], location='right')
        colorbar2.set_label('Amount of fish present')
        #set labels
        ax11.set_title(f'Heatmap of presences of fish on single electrodes\n date: {save_date}')
        ax12.set_xlabel(f'Time in bins [{bin_width}s]')
        fig_single.supylabel('Electrode position [#]')

        fig_comb, (ax21, ax22) = plt.subplots(2,1, sharex=True, sharey=True)
        c21 = ax21.pcolormesh(xc1, yc1, heat_map_matrix_combined_d1, cmap=cmap, shading='auto', vmin=0.1)
        c22 = ax22.pcolormesh(xc2, yc2, heat_map_matrix_combined_d2, cmap=cmap, shading='auto', vmin=0.1)
        colorbar2 = fig_comb.colorbar(c21, ax=[ax21, ax22], location='right')
        colorbar2.set_label('Amount of fish present')
        ax21.set_title(f'Heatmap of presences of fish on combined side by side electrodes\n date: {save_date} ')
        ax22.set_xlabel(f'Time in bins [{bin_width}s]')
        fig_comb.supylabel('Electrode position [#]')

        # Format the x-axis labels to display only hours and minutes
        date_format = mdates.DateFormatter('%H:%M')

        # Formatiere die x-Achse für den ersten Plot
        ax11.xaxis_date()
        ax12.xaxis_date()
        ax11.xaxis.set_major_formatter(date_format)
        ax12.xaxis.set_major_formatter(date_format)

        # Formatiere die x-Achse für den zweiten Plot
        ax21.xaxis_date()
        ax22.xaxis_date()
        ax21.xaxis.set_major_formatter(date_format)
        ax22.xaxis.set_major_formatter(date_format)
        # Save figure
        fig_comb.savefig(f"Heatmap_comb_{bin_width}s_both_dates_compared.png", dpi=600)
        fig_single.savefig(f"Heatmap_single_{bin_width}s_both_dates_compared.png", dpi=600)
        plt.show()
# single days
else:
    # Create a datetime range for the x-axis
    datetimes = [start_date + dt.timedelta(seconds=bin_width * i) for i in range( bin_number + 1)]

    #create 2d MAtrix
    x, y = np.meshgrid(datetimes, np.arange(0, len(electrode_arrangement_list)))
    xc, yc = np.meshgrid(datetimes, np.arange(0, int(len(electrode_arrangement_list) / 2)))

    # Plot the heatmaps with a normalized colormap
    fig_single, ax1 = plt.subplots()
    c1 = ax1.pcolormesh(x, y, heat_map_matrix_single, cmap=cmap, shading='auto', vmin=0.1)
    colorbar1 = fig_single.colorbar(c1, ax=ax1)
    colorbar1.set_label('Amount of fish present')
    ax1.set_title(f'Heatmap of presences of fish on single electrodes\n date: {save_date}')
    ax1.set_xlabel(f'Time in bins [{bin_width}s]')
    ax1.set_ylabel('Electrode position [#]')


    fig_comb, ax2 = plt.subplots()
    c2 = ax2.pcolormesh(xc, yc, heat_map_matrix_combined, cmap=cmap, shading='auto', vmin=0.1)
    colorbar2 = fig_comb.colorbar(c2, ax=ax2)
    colorbar2.set_label('Amount of fish present')
    ax2.set_title(f'Heatmap of presences of fish on combined side by side electrodes\n date: {save_date} ')
    ax2.set_xlabel(f'Time in bins [{bin_width}s]')
    ax2.set_ylabel('Electrode position [#]')

    # Format the x-axis labels to display only hours and minutes
    date_format = mdates.DateFormatter('%H:%M')
    ax1.xaxis_date()
    ax2.xaxis_date()
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    #save figs
    fig_comb.savefig(f"Heatmap_comb_{bin_width}s_{save_date}.png", dpi=600)
    fig_single.savefig(f"Heatmap_single_{bin_width}s_{save_date}.png", dpi=600)
    plt.show()