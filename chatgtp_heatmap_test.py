import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os
import datetime as dt
import random as rn
from IPython import embed
import statistics as stc
from tqdm import tqdm
import math
# load own functions
from fish_list_unpacker import fish_list_unpacker as flu

# auswahl datensatz a=0 21.10.19 a=1 22.10.19
a = 0  # select record day
# auswahl heat_matrix already calculated no = 0, yes=1
b = 0




# save path
load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'

# histogram settings
bin_width = 10  # in seconds

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
from recording_grid_columbia_2019 import x_grid
from recording_grid_columbia_2019 import y_grid
from recording_grid_columbia_2019 import electrode_id_list
##### import fish data:#######


filename = sorted(os.listdir(load_path))[a]
fish_list = np.load(load_path + '/' + filename + '/fishlist.npy', allow_pickle=True)
filepath = load_path + '/' + filename

#arangement of elektrodes from bottomto top
electrode_arrangement_list = [25,17,26,18,27,19,28,20,29,21,30,22,31,23,32,24,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8]


#presence_af = np.load(filepath + '/presence_all_ fish' + str(bin_width) +'.npy')

dic = flu(fish_list, filepath)

ident_af = dic['ident_list']
freq_af = dic['frequence_list']
time_af = dic['time_list']
time_idx_af = dic['time_idx_list']
ch_af = dic['channel_list']
sign_af =dic['sign_list']
times = dic['record_time_array']


    #bin variable
bin_number = math.ceil(max(times)/bin_width)
start_bin = 0
end_bin = bin_width

if b == 0:
    # raw, time in 60s bins, column: all elektrodes from one and to another, numbers of fish that were
    # measured at elektrode in 60s, individual fish onlycounted once

    heat_map_matrix_single = np.zeros((len(electrode_arrangement_list), bin_number), dtype=int)
    heat_map_matrix_combined = np.zeros((len(electrode_arrangement_list) // 2, bin_number), dtype=int)

    for bin in tqdm(range(bin_number)):
        for fish_nr in range(len(time_af)):
            presence = (time_af[fish_nr] >= start_bin + bin * bin_width) & (
                        time_af[fish_nr] <= end_bin + bin * bin_width)
            hits = np.where(presence)[0]
            for hit in hits:

                electrode_idx = np.where(np.array(electrode_arrangement_list) == ch_af[fish_nr][hit]+1)[0]
                heat_map_matrix_single[electrode_idx, bin] += 1
                heat_map_matrix_combined[electrode_idx // 2, bin] += 1

    np.save((filepath+'/heatmap_matrix_single_'+ str(bin_width) +'.npy'),heat_map_matrix_single)
    np.save((filepath + '/heatmap_matrix_combined_' + str(bin_width) +'.npy'), heat_map_matrix_combined)

if b == 1:
    heat_map_matrix_single = np.load(filepath + '/heatmap_matrix_single_' + str(bin_width) + '.npy')
    heat_map_matrix_combined = np.load(filepath + '/heatmap_matrix_combined_' + str(bin_width) + '.npy')



#create mashgrid
x, y = np.mgrid[0:bin_number, 0:len(electrode_arrangement_list)]
xc, yc = np.mgrid[0:bin_number, 0:len(electrode_arrangement_list)//2]

#plot heatmap
fig_single,ax1 =plt.subplots()
c1 = ax1.pcolormesh(x,y,heat_map_matrix_single, cmap='plasma')
fig_single.colorbar(c1,ax=ax1)
ax1.set_title('Heatmap of presences of fish on single electrodes')
ax1.set_xlabel('Time in bins [10s] ')
ax1.set_ylabel('Amount of fish present [#]')
ax1.set_xlim(1520,1600)
# when saving, specify the DPI
fig_single.savefig("Heatmap_single_10s.png", dpi = 600)

#plot heatmap
fig_comb,ax2 = plt.subplots()
c2=ax2.pcolormesh(xc,yc,heat_map_matrix_combined,cmap='plasma')
fig_comb.colorbar(c2, ax=ax2)
ax2.set_title('Heatmap of presences of fish on combined side by side electrodes')
ax2.set_xlabel('Time in bins [10s] ')
ax2.set_ylabel('Amount of fish present [#]')
ax2.set_xlim(1520,1600)
# when saving, specify the DPI
fig_comb.savefig("Heatmap_comb_10s.png", dpi = 600)

plt.show()











