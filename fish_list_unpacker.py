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
from IPython import embed

def fish_list_unpacker(fish_list, filepath):
    # unpacking the fish list and loading ident, sign, frequeny, time idx and time for eevry fish

    #load data
    ident = np.load(filepath + '/ident.npy', allow_pickle=True)
    sign = np.load(filepath + '/sign.npy', allow_pickle=True)
    freq = np.load(filepath + '/freq.npy', allow_pickle=True)
    timeidx = np.load(filepath + '/timeidx.npy', allow_pickle=True)
    times = np.load(filepath + '/times.npy', allow_pickle=True)

    # unpack fishlist
    af_dtime_fish = []  # list of time arrays of a fish for all fish
    af_freq_fish = []  # list of frequencies of a fish over all fish
    af_ident = []  # ident list
    af_ch_list = []  # chanel/electrode list for  fish over all fish
    af_timeidx = []  # time index for all fish
    af_sign = [] # sign amplitude for all fish

    for fish_nr in range(len(fish_list)):
        sf_freq = []
        sf_time = []
        sf_ident = []
        sf_ch = []
        sf_tidx = []
        sf_sign = []

        for i in range(len(fish_list[fish_nr])):
            ch = fish_list[fish_nr][i][0]
            ids = fish_list[fish_nr][i][1]
            time_points = times[timeidx[ch][
                ident[ch] == ids]]  # feändert für test for t in range(len(times)):
            # create a list with dt.datetimes for every idx of times of one fish:
            # for t in range(len(time_points)):
            # date_time = start_date + dt.timedelta(seconds=time_points[t])
            # sf_time.append(date_time)

            sf_time.append(time_points)
            give_every_index_ch = np.ones(len(time_points)) * ch
            sf_ch.append(give_every_index_ch)

            fr = freq[ch][ident[ch] == ids]
            sf_freq.append(fr)

            sf_id = ident[ch][ident[ch] == ids]
            sf_ident.append(sf_id)

            st_time_idx = timeidx[ch][ident[ch] == ids]
            sf_tidx.extend(st_time_idx)

            if ch > 15:
                ch_sign = ch - 16
                sign_idx = np.where(ident[ch] == ids)[0]
                st_sign_idx = (lambda sign, ch, sign_idx: [sign[ch][idx][ch_sign] for idx in sign_idx])(sign, ch, sign_idx)
            else:
                ch_sign = ch
                sign_idx = np.where(ident[ch] == ids)[0]
                st_sign_idx = (lambda sign, ch, sign_idx: [sign[ch][idx][ch_sign] for idx in sign_idx])(sign, ch,
                                                                                                        sign_idx)
            sf_sign.extend(st_sign_idx)

        # fish_time = np.concatenate(time_fish)

        if len(sf_ident[0]) > 1:
            sf_ident = np.concatenate(sf_ident)
        af_ident.append(sf_ident)

        sf_freq = np.concatenate(sf_freq)
        af_freq_fish.append(sf_freq)

        sf_time = np.concatenate(sf_time)
        af_dtime_fish.append(sf_time)

        sf_ch = np.concatenate(sf_ch)
        af_ch_list.append(sf_ch)

        af_timeidx.append(sf_tidx)

        af_sign.append(sf_sign)

    # Change dB amplitude into micro Volt
    # caltulacte miV out of dB

    sign_af_microV = [[10 ** (element / 20) * 1000000 for element in sublist] for sublist in af_sign]


    unpacked_fish_list = {
        'ident_list': af_ident,
        'frequence_list': af_freq_fish,
        'time_list': af_dtime_fish,
        'channel_list': af_ch_list,
        'time_idx_list': af_timeidx,
        'sign_list': af_sign,
        'record_time_array': times,
        'sign_list_microV': sign_af_microV
    }
    return unpacked_fish_list

