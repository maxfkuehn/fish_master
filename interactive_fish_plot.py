import os
import numpy as np
from IPython import embed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


#importiert X und Y Koordinaten des Messgrids
from recording_grid_columbia_2019 import x_grid
from recording_grid_columbia_2019 import y_grid

########## loading data ############
#load fishlist
#auswahl datensatz a=0 22.10.19 a=1 23.10.19
a=1 #select record day

filename = sorted(os.listdir('../../../kuehn/data/analysed_data'))[a]
fish_list = np.load('fishlist'+filename+'.npy', allow_pickle=True)
#load all fish list ready fpr plot
af_dtime= np.load('all_fish_datetime'+filename+'.npy', allow_pickle=True)
af_freq=np.load('all_fish_freq'+filename+'.npy', allow_pickle=True)
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
    start_date = start_date_0
elif a == 1:
    start_date = start_date_1

#Plot all fish of a day and make plot clickable, to get postional information

fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
axes = axes.flatten()



for f in range(len(af_dtime)):
    line, = axes[0].plot(af_dtime[f][:], af_freq[f][:], 'o', picker=True, pickradius=1)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S %d-%m-%Y'))
plt.gcf().autofmt_xdate()
#plot the posiotnal grid
axes[1].plot(x_grid,y_grid,'o')


def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)


fig.canvas.mpl_connect('pick_event', onpick)
plt.show()
