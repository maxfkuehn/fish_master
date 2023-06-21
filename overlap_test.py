import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from IPython import embed
import helper_functions as hf
import os
import datetime
import itertools
import matplotlib.gridspec as gridspec
from random import randint
#from params import *y


def plot_overlap_figure(fig, ax, aifl, emptyN1, emptyN2, emptyN3, emptyN4, color_v):
    # plot lines in figure
    for plot_cidx in range(16):
        fish1 = aifl[plot_cidx, emptyN1, ~np.isnan(aifl[plot_cidx, emptyN1])]
        fish2 = aifl[plot_cidx, emptyN2, ~np.isnan(aifl[plot_cidx, emptyN2])]
        for eN1_idx in range(len(fish1)):
            ax.plot(timeidx[plot_cidx][ident[plot_cidx] == fish1[eN1_idx]],
                     freq[plot_cidx][ident[plot_cidx] == fish1[eN1_idx]],
                     Linewidth=3, label='1', color='k', alpha=0.3)
        for eN2_idx in range(len(fish2)):
            ax.plot(timeidx[plot_cidx][ident[plot_cidx] == fish2[eN2_idx]],
                     freq[plot_cidx][ident[plot_cidx] == fish2[eN2_idx]],
                     Linewidth=3, label='2', color='b', alpha=0.3)
        if emptyN3 is not None:
            fish3 = aifl[plot_cidx, emptyN3, ~np.isnan(aifl[plot_cidx, emptyN3])]
            for eN3_idx in range(len(fish3)):
                ax.plot(timeidx[plot_cidx][ident[plot_cidx] == fish3[eN3_idx]],
                         freq[plot_cidx][ident[plot_cidx] == fish3[eN3_idx]],
                         Linewidth=3, label='3', color='green', alpha=0.3)
        if emptyN4 is not None:
            fish4 = aifl[plot_cidx, emptyN4, ~np.isnan(aifl[plot_cidx, emptyN4])]
            for eN4_idx in range(len(fish4)):
                ax.plot(timeidx[plot_cidx][ident[plot_cidx] == fish4[eN4_idx]],
                         freq[plot_cidx][ident[plot_cidx] == fish4[eN4_idx]],
                         Linewidth=3, label='4', color='orange', alpha=0.3)
    # plot the line which has to be assigned to one of the emptyN
    ax.plot(timeidx[channel_idx][ident[channel_idx] == false_fish[ff_idx]],
            freq[channel_idx][ident[channel_idx] == false_fish[ff_idx]],
            Linewidth=1, color='red')

    # create legend without duplicated labels
    hf.legend_without_duplicate_labels(fig, ax)
    plt.show()

def get_list_of_fishN_with_overlap(aifl, fish_in_aifl, time, identity):
    liste = []
    for fishN in fish_in_aifl:
        for Ch_idx in range(16):
            fishs_idxs = aifl[Ch_idx, fishN, ~np.isnan(aifl[Ch_idx, fishN])]
            len_f_idxs = len(fishs_idxs)
            if len_f_idxs >= 2:
                time_traces = []
                for f_idx in range(len_f_idxs):
                    time_traces.append(time[Ch_idx][identity[Ch_idx] == fishs_idxs[f_idx]])

                for subset in itertools.combinations(fishs_idxs, 2):
                    r1 = set(time[Ch_idx][identity[Ch_idx] == subset[0]])
                    result = r1.intersection(time[Ch_idx][identity[Ch_idx] == subset[1]])
                if bool(result):
                    print('overlap -- new sorting')
                    liste.append(fishN)
    liste = np.unique(liste)
    return liste


def assigne_ID_to_fishN_in_aifl(aifl, CH_idx, false_fish, i, empty_counter, fishN, N1, N2, N3, N4):
    if in_str == '1':
        aifl = hf.add_id_to_aifl(aifl, CH_idx, false_fish[i], [[N1]])
    elif in_str == '2':
        aifl = hf.add_id_to_aifl(aifl, CH_idx, false_fish[i], [[N2]])
    elif in_str == '3':
        if N3 is None:
            N3 = int(empty_fishNs[empty_counter])
            empty_counter += 1
            aifl = hf.add_id_to_aifl(aifl, CH_idx, false_fish[i], [[N3]])
            print(fishN, N1, N2, N3)
        else:
            aifl = hf.add_id_to_aifl(aifl, CH_idx, false_fish[i], [[N3]])
    elif in_str == '4':
        if N4 is None:
            N4 = int(empty_fishNs[empty_counter])
            empty_counter += 1
            aifl = hf.add_id_to_aifl(aifl, CH_idx, false_fish[i], [[N4]])
            print(fishN, N1, N2, N3, N4)
        else:
            aifl = hf.add_id_to_aifl(aifl, CH_idx, false_fish[i], [[N4]])
    else:
        print('trace was not assigned to any fish -- trace Ch+ID:', CH_idx, false_fish[i])

    return aifl, empty_counter, N3, N4


if __name__ == '__main__':

    ###################################################################################################################
    # load data
    ###################################################################################################################
    # load all the data of one day
    filename= sorted(os.listdir('../../../kuehn/data/'))[1]

    ident = np.load('../../../kuehn/data/' + filename + '/ident.npy',
                          allow_pickle=True)
    freq = np.load('../../../kuehn/data/' + filename + '/freq.npy',
                         allow_pickle=True)
    timeidx = np.load('../../../kuehn/data/' + filename + '/timeidx.npy',
                            allow_pickle=True)

    aifl = np.load('../../../kuehn/data/aifl.npy', allow_pickle=True)
    faifl = np.load('../../../kuehn/data/faifl.npy', allow_pickle=True)
    # faifl = np.delete(faifl, [0], axis=0)

    ###################################################################################################################
    # params
    # color_v, font_size, _, _, _, _ = params.params()
    color_vec=[]
    fs = 12

    for i in range(10):
        color_vec.append('#%06X' % randint(0, 0xFFFFFF))
     # variables
    empty_counter = 0

    # lists
    fish_in_aifl = list(np.unique(np.where(~np.isnan(aifl[:, :, 0]))[1]))
    fish_in_faifl = list(np.unique(faifl[:, [1, 3]]))
    ###################################################################################################################
    # get me the fish_numbers with overlapping traces
    new_sorting = get_list_of_fishN_with_overlap(aifl, fish_in_aifl, timeidx, ident)

    ###################################################################################################################
    # get me the fish_numbers with no fish_ids
    empty_fishNs = []
    for i in range(len(aifl[0])):
        if np.all(np.isnan(aifl[:, i])):
            empty_fishNs = np.append(empty_fishNs, i)

    ###################################################################################################################
    # get me the fish_numbers with no fish_ids
    for fish_number in fish_in_aifl:

        emptyN3 = None
        emptyN4 = None
        ax = hf.plot_together(timeidx, freq, ident, aifl, int(fish_number), color_vec[0])

        if fish_number not in new_sorting:
            correct_str = input('Do we have to correct the fish? [y/n]')
        else:
            correct_str = 'y'

        if correct_str == 'n':
            continue
        else:
            emptyN1 = int(empty_fishNs[empty_counter])
            empty_counter += 1
            emptyN2 = int(empty_fishNs[empty_counter])
            empty_counter += 1
            print(fish_number, emptyN1, emptyN2)
            for channel_idx in range(32):
                false_fish = aifl[channel_idx, int(fish_number), ~np.isnan(aifl[channel_idx, int(fish_number)])]
                for ff_idx in range(len(false_fish)):
                    # __________________________________________________________________________________________________
                    # make figure
                    fig1 = plt.figure(figsize=(16, 14))
                    gs = gridspec.GridSpec(1, 1, left=0.125, right=0.95, top=0.90, bottom=0.15, wspace=0.15, hspace=0.15)
                    ax1 = fig1.add_subplot(gs[0, 0])
                    ax1.set_xlim(ax.get_xlim())
                    ax1.set_ylim(ax.get_ylim())
                    fig1.suptitle('Ch {}, ID {}'.format(channel_idx, false_fish[ff_idx]), fontsize=fs)

                    for i in range(32):
                        fish1 = aifl[i, fish_number, ~np.isnan(aifl[i, fish_number])]
                        r1 = len(fish1)
                        # print(fish1)
                        for len_idx1 in range(r1):
                            embed()
                            quit()
                            plt.plot(timeidx[i][ident[i] == fish1[len_idx1]],
                                     freq[i][ident[i] == fish1[len_idx1]],
                                     Linewidth=1, color='gray', alpha=0.2)

                    plot_overlap_figure(fig1, ax1, aifl, emptyN1, emptyN2, emptyN3, emptyN4, ['k'])
                    # __________________________________________________________________________________________________
                    # where does the new red line belong to? -- then fill aifl at the fish_number with the ID of the red
                    # line
                    in_str = input('Where does the traces belong to? [1/2/3/4]')

                    aifl, empty_counter, emptyN3, emptyN4 = assigne_ID_to_fishN_in_aifl(aifl, channel_idx, false_fish,
                                                                                        ff_idx, empty_counter,
                                                                                        fish_number, emptyN1, emptyN2,
                                                                                        emptyN3, emptyN4)

            aifl[:, fish_number, :] = np.nan
    ###################################################################################################################
    # save
    if filename not in os.listdir('../../../kuehn/data/'):
        os.mkdir('../../../kuehn/data/'+filename)

    np.save('.../../../kuehn/data/' + filename + '/aifl2.npy', aifl)
    np.save('../../../kuehn/data/' + filename + '/faifl2.npy', faifl)

    embed()
    quit()
