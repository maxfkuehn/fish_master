import os
import numpy as np
from IPython import embed

from tqdm import tqdm
import matplotlib.pyplot as plt
from random import random
import sys
import matplotlib.dates as mdates
import datetime as dt

########## loading data ############
#load fishlist
#auswahl datensatz a=0  a=1
#21.10.19
filename_0 = sorted(os.listdir('../../../kuehn/data/analysed_data'))[0]
fish_list_0 = np.load('fishlist'+filename+'.npy', allow_pickle=True)

#22.10.19
filename_1= sorted(os.listdir('../../../kuehn/data/analysed_data'))[1]
fish_list_1 = np.load('fishlist'+filename+'.npy', allow_pickle=True)

    fish_freq = np.concatenate(freq_fish)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S %d-%m-%Y'))
    #plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(time_fish,fish_freq,'o')
    plt.gcf().autofmt_xdate()

plt.show()