import datetime as dt
import math
import os
import random as rn
import statistics as stc
import seaborn as sns
import pdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from IPython import embed
from tqdm import tqdm
from scipy.signal import savgol_filter
# load own functions
from fish_list_unpacker import fish_list_unpacker as flu
#### importiert electrode position grid ####
from recording_grid_columbia_2019 import x_grid
from recording_grid_columbia_2019 import y_grid
from recording_grid_columbia_2019 import electrode_id_list
from recording_grid_columbia_2019 import electrode_arrangement_list
from grid_middle_line_calculator import grid_middle_line_calculator as gmlc
import random
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from colorcet.plotting import swatch, swatches, candy_buttons

#### date time ######
# definiere Zeitpunkte der Datei als Datum (datetime):
start_date_0 = dt.datetime(2019, 10, 21, 13, 25, 00)
start_date_1 = dt.datetime(2019, 10, 22, 8, 13, 00)
record_day_0 = '/2019-10-21-13_25'
record_day_1 = '/2019-10-22-8_13'
# load path
load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'
# load data

sunrise_day1 = dt.datetime(2019, 10, 21, 5, 34, 00)
sunset_day1 = dt.datetime(2019, 10, 21, 17, 54, 00)
sunrise_day2 = dt.datetime(2019, 10, 22, 5, 34, 00)
sunset_day2 = dt.datetime(2019, 10, 22, 17, 54, 00)
sunrise_day3 = dt.datetime(2019, 10, 23, 5, 34, 00)

end_recording_1 = start_date_0 + dt.timedelta(seconds=66850)


both_days_time = []
both_days_species = []
both_days_freq = []
both_days_movement = []

frq_list = []
time_dt = []


for a in [0,1]:
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

    time = np.load(load_path + record_day + '/timepoints_smooth_movement.npy', allow_pickle=True)
    move = np.load(load_path + record_day + '/smoothed_fish_movement.npy' , allow_pickle=True)
    freq = np.load(load_path + record_day + '/freq_values_movement.npy' , allow_pickle=True)
    species = np.load(load_path + record_day + '/species_list_corrected.npy' , allow_pickle=True)

    both_days_time.append(time)
    both_days_species.append(species)
    both_days_freq.append(freq)
    both_days_movement.append(move)

    for b in range(len(time)):
        freq_kaputt = freq[b]
        freq_fish = [f[0] for f in freq_kaputt]
        frq_list.append(freq_fish)
        dt_time = [start_date + dt.timedelta(seconds=t) for t in time[b]]
        time_dt.append(dt_time)

time_list = np.concatenate(both_days_time)
movement_list =np.concatenate(both_days_movement)
species_list=np.concatenate(both_days_species)

eigenmannia = 'Eigenmania'
apteronotus = 'Apteronotus'

embed()
eig = np.where(species_list == eigenmannia)[0]
apt = np.where(species_list == apteronotus)[0]

apt_mf = []
apt_ff = []

frq_apt = [frq_list[x] for x in apt]
frq_eig = [frq_list[x] for x in eig]

for f in frq_apt:
    a =max(f)- min(f)
    if f[0]>800:
        apt_mf.append(a)
    else:
        apt_ff.append(a)

eig_f = []
for f in frq_eig:
    a =max(f)- min(f)
    eig_f.append(a)


stats.ttest_ind(a=eig_f, b=apt_ff)
stats.ttest_ind(a=eig_f, b=apt_mf)

min_range =[[40000],[38000]]
max_range = [[54000],[69000]]

for i in [0,1]:
    fig,(ax1,ax2) = plt.subplots(1,2)
    for b in range(len(both_days_time[i])):
        freq_kaputt = both_days_freq[i][b]
        freq_fish = [f[0] for f in freq_kaputt]
        if any((min_range[i] < num < max_range[i] for num in np.array(both_days_time[i][b]))):
            ax2.plot(both_days_time[i][b],both_days_movement[i][b],'o',label = f'{b}')
            ax1.plot(both_days_time[i][b],freq_fish,'o',label = f'{b}')
    plt.legend()


#create figure
all_fish_freq_time_plot, ax1 = plt.subplots(figsize=(9,6))
# colorpalette


#plot frequenz verteilung
for idx in range(len(frq_list)):
    ax1.plot(time_dt[idx], frq_list[idx],'o', zorder=1)

# day nigth and recording stop
ax1.axvspan(end_recording_1, start_date_1, facecolor='black', zorder=2)
ax1.axvspan(sunset_day1,sunrise_day2,  facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2,sunrise_day3,  facecolor='black', alpha=0.2)

#set label
ax1.set_xlabel('Time',fontsize = 16)
ax1.set_ylabel('Frequency [Hz]', fontsize = 16)

# date time einstellungen
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
#save fig
all_fish_freq_time_plot.savefig('F-T_bothdays.png', dpi=300)

''' 
New Figure:
short migration,transit
'''


#create figure
plot_transit_short,ax1 = plt.subplots()

#plot frequenz verteilung

label = int(np.mean(frq_list[17]))
times = time_dt[17]
species = species_list[17]
ax1.plot(times,np.array(movement_list[17])/100,'o-',label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1,sunrise_day2,  facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2,sunrise_day3,  facecolor='black', alpha=0.2)
ax2.axvspan(sunset_day1,sunrise_day2,  facecolor='black', alpha=0.2)
ax2.axvspan(sunset_day2,sunrise_day3,  facecolor='black', alpha=0.2)

#set label
ax1.set_xlabel('Time',fontsize = 16)
ax1.set_ylabel('Distance [m]', fontsize = 16)
#set lims
ax1.set_ylim(0,16)
times_diff= (times[-1]-times[0]).total_seconds()*0.02
ax1.set_xlim(times[0]-dt.timedelta(seconds=times_diff),times[-1]+dt.timedelta(seconds=times_diff))
#ax1.set_xlabel()
# date time einstellungen
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
#show legend
ax1.legend()
#save fig
plot_transit_short.savefig('Plot_transit_short.png', dpi=300)

''' 
New Figure:
bsp. visit
'''
#create figure
bsp_short_migration,ax1 = plt.subplots()

#plot frequenz verteilung

label = int(np.mean(frq_list[148]))
times = time_dt[148]
species = species_list[17]
ax1.plot(times,np.array(movement_list[148])/100,'o-',label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1,sunrise_day2,  facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2,sunrise_day3,  facecolor='black', alpha=0.2)
ax2.axvspan(sunset_day1,sunrise_day2,  facecolor='black', alpha=0.2)
ax2.axvspan(sunset_day2,sunrise_day3,  facecolor='black', alpha=0.2)

#set label
ax1.set_xlabel('Time',fontsize = 16)
ax1.set_ylabel('Distance [m]', fontsize = 16)
#set lims
ax1.set_ylim(0,16)
times_diff= (times[-1]-times[0]).total_seconds()*0.02
ax1.set_xlim(times[0]-dt.timedelta(seconds=times_diff),times[-1]+dt.timedelta(seconds=times_diff))
# date time einstellungen
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
#show legend
ax1.legend()
#save fig
bsp_short_migration.savefig('lel.png', dpi=300)

''' 
New Figure:
bsp. visit
'''
#create figure
plot1,ax1 = plt.subplots()

#plot frequenz verteilung

label = int(np.mean(frq_list[0]))
times = time_dt[0]
species = species_list[0]
ax1.plot(time_dt[0],np.array(movement_list[0])/100,'o-',label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1,sunrise_day2,  facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2,sunrise_day3,  facecolor='black', alpha=0.2)
ax2.axvspan(sunset_day1,sunrise_day2,  facecolor='black', alpha=0.2)
ax2.axvspan(sunset_day2,sunrise_day3,  facecolor='black', alpha=0.2)

#set label
ax1.set_xlabel('Time',fontsize = 16)
ax1.set_ylabel('Distance [m]', fontsize = 16)
#set lims
ax1.set_ylim(0,16)

times_diff= (times[-1]-times[0]).total_seconds()*0.02
ax1.set_xlim(times[0]-dt.timedelta(seconds=times_diff),times[-1]+dt.timedelta(seconds=times_diff))
# date time einstellungen
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
#show legend
ax1.legend()

#save fig
plot1.savefig('plot1', dpi=300)


''' 
New Figure:
bsp. visit
'''

plot2, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[6]))
times = time_dt[6]
species = species_list[6]
ax1.plot(time_dt[6], np.array(movement_list[6]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot2.savefig('plot2', dpi=300)


''' 
New Figure:
bsp. visit
'''

plot3, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[7]))
times = time_dt[7]
species = species_list[7]
ax1.plot(time_dt[7], np.array(movement_list[7]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot3.savefig('plot3', dpi=300)


''' 
New Figure:
bsp. visit
'''

plot4, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[17]))
times = time_dt[17]
species = species_list[17]
ax1.plot(time_dt[17], np.array(movement_list[17]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()


''' 
New Figure:
bsp. visit
'''

plot5, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[24]))
times = time_dt[24]
species = species_list[24]
ax1.plot(time_dt[24], np.array(movement_list[24]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot5.savefig('plot5', dpi=300)
# save fig
plot4.savefig('plot4', dpi=300)


''' 
New Figure:
bsp. visit
'''

plot6, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[33]))
times = time_dt[33]
species = species_list[33]
ax1.plot(time_dt[33], np.array(movement_list[33]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

''' 
New Figure:
bsp. visit
'''

# save fig
plot6.savefig('plot6', dpi=300)

plot7, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[39]))
times = time_dt[39]
species = species_list[39]
ax1.plot(time_dt[39], np.array(movement_list[39]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot7.savefig('plot7', dpi=300)


''' 
New Figure:
bsp. visit
'''

plot8, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[55]))
times = time_dt[55]
species = species_list[55]
ax1.plot(time_dt[55], np.array(movement_list[55]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot8.savefig('plot8', dpi=300)


''' 
New Figure:
bsp. visit
'''

plot9, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[57]))
times = time_dt[57]
species = species_list[57]
ax1.plot(time_dt[57], np.array(movement_list[57]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot9.savefig('plot9', dpi=300)

''' 
New Figure:
bsp. visit
'''

plot10, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[108]))
times = time_dt[108]
species = species_list[108]
ax1.plot(time_dt[108], np.array(movement_list[108]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot10.savefig('plot10', dpi=300)

''' 
New Figure:
bsp. visit
'''

plot11, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[84]))
times = time_dt[84]
species = species_list[84]
ax1.plot(time_dt[84], np.array(movement_list[84]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot11.savefig('plot11', dpi=300)

''' 
New Figure:
bsp. visit
'''

plot12, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[237]))
times = time_dt[237]
species = species_list[237]
ax1.plot(time_dt[237], np.array(movement_list[237]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot12.savefig('plot13', dpi=300)

''' 
New Figure:
bsp. visit
'''
plot13, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[254]))
times = time_dt[254]
species = species_list[254]
ax1.plot(time_dt[254], np.array(movement_list[254]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
plot13.savefig('plot13', dpi=300)


''' 
New Figure:
bsp. visit
'''
stayed1, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[68]))
times = time_dt[68]
species = species_list[68]
ax1.plot(time_dt[68], np.array(movement_list[68]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
stayed1.savefig('plot_stayed1', dpi=300)

''' 
New Figure:
bsp. visit
'''
stayed2, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[101]))
times = time_dt[101]
species = species_list[101]
ax1.plot(time_dt[101], np.array(movement_list[101]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
stayed2.savefig('plot_stayed2', dpi=300)


''' 
New Figure:
bsp. visit
'''
stayed3, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[75]))
times = time_dt[75]
species = species_list[75]
ax1.plot(time_dt[75], np.array(movement_list[75]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
stayed3.savefig('plot_stayed3', dpi=300)


''' 
New Figure:
bsp. visit
'''
stayed4, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[164]))
times = time_dt[164]
species = species_list[164]
ax1.plot(time_dt[164], np.array(movement_list[164]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig
stayed4.savefig('plot_stayed4', dpi=300)

''' 
New Figure:
bsp. visit
'''
duration, ax1 = plt.subplots()
# plot frequenz verteilung

label = int(np.mean(frq_list[68]))
times = time_dt[68]
species = species_list[68]
ax1.plot(time_dt[68], np.array(movement_list[68]) / 100, 'o-', label=f'{species}:\n{label} Hz')

# day nigth and recording stop

ax1.axvspan(sunset_day1, sunrise_day2, facecolor='black', alpha=0.2)
ax1.axvspan(sunset_day2, sunrise_day3, facecolor='black', alpha=0.2)

# set label
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Distance [m]', fontsize=16)
# set lims
ax1.set_ylim(0, 16)

times_diff = (times[-1] - times[0]).total_seconds() * 0.05
ax1.set_xlim(times[0] - dt.timedelta(seconds=times_diff), times[-1] + dt.timedelta(seconds=times_diff))
# date time einstellungen
if times_diff / 0.02 > 300:
    date_format = mdates.DateFormatter('%H:%M')
else:
    date_format = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(date_format)
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# show legend
ax1.legend()

# save fig

duration.savefig('männchen_lang', dpi=300)

plt.close()

hister, (ax1,ax2)= plt.subplots(1,2,figsize=(9,6))

long_diff= []
for t in time_list:
     l=max(t)-min(t)
     long_diff.append(l)

diff = np.array(long_diff)

apt_dif = diff[apt]/60/60
eig_dif = diff[eig]/60/60

bin_width = 300/60/60

data_range = max(max(apt_dif), max(eig_dif))


num_bins = int(data_range / bin_width)

ax1.hist(apt_dif,bins=num_bins, edgecolor = 'red',label='Aptoronotus sp.',histtype='stepfilled', color='red')
ax2.hist(eig_dif,bins=num_bins, edgecolor = 'blue',label='Eigenmannia sp.',histtype='stepfilled', color='blue')
# fontsize x and y tick
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# set label
hister.supxlabel('Time [hours]', fontsize=16)
ax1.set_ylabel('Count', fontsize=16)
ax2.set_ylabel([])

# show legend
ax1.legend()
ax2.legend()
hister.savefig('hist_duration', dpi=300)


#### statisic duration tag nacht
sunset_day1 = dt.datetime(2019, 10, 21, 17, 54, 00)
sunrise_day2 = dt.datetime(2019, 10, 22, 5, 34, 00)

start1 = (sunset_day1-start_date_0).total_seconds()+600
ende1 = (sunrise_day2-start_date_0).total_seconds()-3600
start2 = (sunset_day2-start_date_0).total_seconds()+300
ende2 = (sunrise_day3-start_date_0).total_seconds()-600

apt_tag= []
apt_nacht = []
eig_tag = []
eig_nacht= []

for i in time_list[apt]:
    if start1<i[0]<ende1 or start2<i[0]<ende2:
        ad= max(i)-min(i)
        apt_nacht.append(ad)
    else:
        da =max(i)-min(i)
        apt_tag.append(da)


for i in time_list[eig]:
    if start1 < i[0] < ende1 or start2 < i[0] < ende2:
        ad= max(i)-min(i)
        eig_nacht.append(ad)
    else:
        da =max(i)-min(i)
        eig_tag.append(da)

embed()
plt.show()