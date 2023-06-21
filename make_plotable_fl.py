import os
import numpy as np
from IPython import embed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

#make a fishlist with for frq and date time of everyfish, to be plotable. This data is used for plotting and
# working with its aswell

########## loading data ############
#load fishlist
#auswahl datensatz a=0 22.10.19 a=1 23.10.19
a=1
filename = sorted(os.listdir('../../../kuehn/data/analysed_data'))[a]
fish_list = np.load('fishlist'+filename+'.npy', allow_pickle=True)

#load recorded data
filepath =('../../../kuehn/data/analysed_data/' + filename)
ident = np.load(filepath + '/ident.npy', allow_pickle=True)
sign = np.load(filepath + '/sign.npy', allow_pickle=True)
freq = np.load(filepath + '/freq.npy', allow_pickle=True)
timeidx = np.load(filepath + '/timeidx.npy', allow_pickle=True)
times = np.load(filepath + '/times.npy', allow_pickle=True)


#date data #
#definiere Zeitpunkte der Datei als Datum (datetime):
start_date_0 = dt.datetime(2019, 10, 21, 13, 25, 00 )
start_date_1 = dt.datetime(2019, 10, 22, 8, 13, 00 )

if a == 0:
    start_date=start_date_0
elif a == 1:
    start_date = start_date_1

## Schleife über alle Fische in der Fisch Listfish

# create date_time and frequ list of every single fish ia fish list


af_dtime_fish = []
af_freq_fish = []
af_ch_list = []
for fish_nr in range(len(fish_list)):
    sf_freq = []
    sf_time = []
    sf_ch = []
    for i in range(len(fish_list[fish_nr])):
        ch = fish_list[fish_nr][i][0]
        ids = fish_list[fish_nr][i][1]
        time_points = times[timeidx[ch][ident[ch] == ids]]  # feändert für test for t in range(len(times)):
        # create a list with with dt.datetimes for every idx of times of one fish:
        for t in range(len(time_points)):
            date_time = start_date + dt.timedelta(seconds=time_points[t])
            sf_time.append(date_time)

        fr = freq[ch][ident[ch] == ids]

        sf_ch.append(ch)

        sf_freq.append(fr)

    # fish_time = np.concatenate(time_fish)
    sf_freq = np.concatenate(sf_freq)
    af_freq_fish.append(sf_freq)

    af_dtime_fish.append(sf_time)


    af_ch_list.append(sf_ch)


x1 = af_dtime_fish[0][10]
x2 = af_dtime_fish[5][-1]
y1 = 500
y2 = 1000
X = []




embed()
np.save('all_fish_datetime'+filename+'.npy', af_dtime_fish)
np.save('all_fish_freq'+filename+'.npy', af_freq_fish)

