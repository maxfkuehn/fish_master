import os
import numpy as np
from IPython import embed
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import random
import sys
import matplotlib.dates as mdates
import datetime as dt
import math

########## loading data ############
#load fishlist
filename = sorted(os.listdir('../../../kuehn/data/analysed_data'))[0]
fish_list = np.load('fishlist'+filename+'.npy', allow_pickle=True)

#load recorded data
filepath =('../../../kuehn/data/analysed_data/' + filename)
ident = np.load(filepath + '/ident.npy', allow_pickle=True)
sign = np.load(filepath + '/sign.npy', allow_pickle=True)
freq = np.load(filepath + '/freq.npy', allow_pickle=True)
timeidx = np.load(filepath + '/timeidx.npy', allow_pickle=True)
times = np.load(filepath + '/times.npy', allow_pickle=True)

# Number of max fish amount measured
fish_count = len(fish_list)

#looping over all fish
for fish_nr in range(fish_count):
    #get freq and time for each fish
    time_fish = []
    freq_fish = []

    for i in range(len(fish_list[fish_nr])):
        ch = fish_list[fish_nr][i][0]
        ids = fish_list[fish_nr][i][1]
        t = times[timeidx[ch][ident[ch] == ids]]
        fr = freq[ch][ident[ch] == ids]

        time_fish.append(t)
        freq_fish.append(fr)


embed()