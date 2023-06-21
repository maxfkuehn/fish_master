import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from IPython import embed


# fill aifl with the IDs
def fill_aifl(id0, id1, aifl, Ch, Ch_connect, matrix_fish_counter, time, frequency, identity, faifl):
    """ this function checks where to adds the two identities to the aifl

    3 cases:
    1: both ids do not exist in aifl - new fish and both ids are added to that fish number
    2: both ids already exist in aifl - check whether the ids exist in the same fish number -- old fish
                                      - if not the same number the traces of both fish numbers are plotted and than
                                      by input of the person either but together as on fish number or added to the faifl
    3: one of the ids exist in aifl - add the other id to the same fish number

    Parameters
    ----------
    id0:        int
                ID of the first trace
    id1:        int
                ID of the second trace
    aifl:       3-D array
                all identity fish list; 1_D: channel;    2_D: fish number;      3_D: fish identities
    Ch:         int
                first channel
    Ch_connect: int
                channel of id1 and to which we connect
    matrix_fish_counter: int
                counter where the new fish should be registered
    time:       2-D array
                loaded time array; 1_D: channel;         2_D: arbitrary time points [int]
    frequency:  2-D array
                loaded frequency array; 1_D: channel;    2_D: frequencies [Hz]
    identity:   2-D array
                loaded ident array; 1_D: channel;        2_D: identities [int]
    faifl:      2-D array
                false all identity fish list; 1_D: number of how many false enteries in the aifl exist
                2_D: channels and fish number that are falsely connected;
                4 columns: Ch, fishN0, Ch_connect, fishN1

    Returns
    -------
    matrix_fish_counter: int
                updated counter
    aifl:       3-D array
                updated aifl
    faifl       2-D array
                updated faifl
    """

    # fish number
    fishN0 = np.where(aifl[Ch][:][:] == id0)
    fishN1 = np.where(aifl[Ch_connect][:][:] == id1)

    # when none of the IDs existes -- new fish
    if aifl[Ch][fishN0].size == 0 and aifl[Ch_connect][fishN1].size == 0:
        aifl[Ch][matrix_fish_counter][0] = id0
        aifl[Ch_connect][matrix_fish_counter][0] = id1
        matrix_fish_counter += 1
        print('new fish - ', matrix_fish_counter-1)

    # what happens when both IDs already exist in the channels
    elif aifl[Ch][fishN0].size != 0 and aifl[Ch_connect][fishN1].size != 0:
        try:
            # both IDs are already together as one fish
            if fishN0[0] == fishN1[0]:
                print('old fish')
            # the IDs are in two different fish, we have to check is the fish should be merged
            # plotting to identify by eye
            # then input if the traces should be merged:
            # yes/no/false --- if the merge before was already false
            else:
                print('confused fish', fishN0[0][0], fishN1[0][0])

                a1 = aifl[:, fishN0[0], :]
                a2 = aifl[:, fishN1[0], :]

                for i in range(a1.shape[0]):
                    if a2[i, 0, ~np.isnan(a2[i, 0, :])].size != 0:
                        append_counter = 1
                        for j in range(a2[i, 0, ~np.isnan(a2[i, 0, :])].size):
                            print(a2[i][0][j])
                            nan_pos = np.where(a1[i, 0, ~np.isnan(a1[i, 0, :])])[0]
                            if nan_pos.size != 0:

                                aifl[i, fishN0[0], nan_pos[-1] + append_counter] = a2[i, 0, j]
                                append_counter += 1
                            else:
                                aifl[i, fishN0[0], 0] = a2[i, 0, j]
                aifl[:, fishN1[0], :] = np.nan

                # plot_confused_fish(time, frequency, identity, aifl, id0, id1, Ch, Ch_connect)
                # go_signal = input('Are the traces matching? [y/n/f]')
                #
                # # if the merge before was already false and the fish_number is already in the faifl
                # if np.any(faifl[:, [1, 3]] == fishN0[0][0]) or np.any(faifl[:, [1, 3]] == fishN1[0][0]):
                #     print('Traces are not matching')
                #     faifl = np.append(faifl, [[Ch, fishN0[0][0], Ch_connect, fishN1[0][0]]], axis=0)
                # # go_signal = yes: merge the two fish to one
                # elif go_signal == 'y':
                #     a1 = aifl[:, fishN0[0], :]
                #     a2 = aifl[:, fishN1[0], :]
                #
                #     for i in range(a1.shape[0]):
                #         if a2[i, 0, ~np.isnan(a2[i, 0, :])].size != 0:
                #             append_counter = 1
                #             for j in range(a2[i, 0, ~np.isnan(a2[i, 0, :])].size):
                #                 print(a2[i][0][j])
                #                 nan_pos = np.where(a1[i, 0, ~np.isnan(a1[i, 0, :])])[0]
                #                 if nan_pos.size != 0:
                #                     aifl[i, fishN0[0], nan_pos[-1] + append_counter] = a2[i, 0, j]
                #                     append_counter += 1
                #                 else:
                #                     aifl[i, fishN0[0], 0] = a2[i, 0, j]
                #     aifl[:, fishN1[0], :] = np.nan
                #
                # # go_signal = false: do not merge and put the fish_number into the faifl
                # elif go_signal == 'f':
                #     faifl = np.append(faifl, [[Ch, fishN0[0][0], Ch_connect, fishN1[0][0]]], axis=0)
                #
                # # go_signal = everything else/no: no merge
                # else:
                #     print('no merge')
        except:
            embed()
            quit()

    # if one of the fish does exist but the other one not:
    # if fish0 exists assign fish1 the same fish_number
    elif aifl[Ch][fishN0].size != 0:
        aifl = add_id_to_aifl(aifl, Ch_connect, id1, fishN0)

    # if fish1 exists assign fish0 the same fish_number
    elif aifl[Ch_connect][fishN1].size != 0:
        aifl = add_id_to_aifl(aifl, Ch, id0, fishN1)

    return matrix_fish_counter, aifl, faifl


def add_id_to_aifl(aifl, Ch, ID, fishN):
    """ adds the ID to the fishN in aifl

    Parameters
    ----------
    aifl:       3-D array
                all identity fish list; 1_D: channel;    2_D: fish number;      3_D: fish identities
    Ch:         int
                Channel
    ID:         int
                the fishID which is added to aifl
    fishN:      int
                the fish number to which we add the ID

    Returns
    -------
    aifl:       3-D array
                all identity fish list; 1_D: channel;    2_D: fish number;      3_D: fish identities
                with the new ID at the fishN
    """

    nan_pos = np.where(aifl[Ch][fishN[0]][~np.isnan(aifl[Ch][fishN[0]])])[0]
    if nan_pos.size != 0:
        aifl[Ch][fishN[0][0]][nan_pos[-1] + 1] = ID
    else:
        aifl[Ch][fishN[0][0]][0] = ID
    return aifl


def plot_confused_fish(time, frequency, identity, aifl, id0, id1, Ch, Ch_next):
    """ plots the two traces which should be connected but already exists in two different fish numbers in the aifl
    it plots all the traces of this channel and the next of both fish numbers

    plot both traces in question: fish0 -- red;  fish1 -- blue
    plot the existing traces of the fish_numbers in question
    plot traces of fish0 in channel and channel+1 in grey and black
    plot traces of fish1 in channel and channel+1 in green and yellow

    Parameters
    ----------
    time:       2-D array
                loaded time array; 1_D: channel;         2_D: arbitrary time points [int]
    frequency:  2-D array
                loaded frequency array; 1_D: channel;    2_D: frequencies [Hz]
    identity:   2-D array
                loaded ident array; 1_D: channel;        2_D: identities [int]
    aifl:       3-D array
                all identity fish list; 1_D: channel;    2_D: fish number;      3_D: fish identities
    id0:        int
                identity of the first trace
    id1:        int
                identity of the second trace
    Ch:    int
                current channel
    Ch_next: int
                next channel to which we connect

    Returns
    -------
    nothing

    """
    # parameter
    t0 = time[Ch][identity[Ch] == id0]
    f0 = frequency[Ch][identity[Ch] == id0]
    t1 = time[Ch_next][identity[Ch_next] == id1]
    f1 = frequency[Ch_next][identity[Ch_next] == id1]
    fishN0 = np.where(aifl[Ch][:][:] == id0)
    fishN1 = np.where(aifl[Ch_next][:][:] == id1)

    # ToDo: ich hab das == True entfernt, wenns nicht mehr läuft zurück ändern
    #  ~np.isnan(aifl[connect_channel][where_id1[0]]) == True
    for plotidx0 in range(len(np.where(~np.isnan(aifl[Ch][fishN0[0]]))[1])):
        plt.plot(time[Ch][identity[Ch] == aifl[Ch][fishN0[0][0]][plotidx0]],
                 frequency[Ch][identity[Ch] == aifl[Ch][fishN0[0][0]][plotidx0]],
                 'grey', LineWidth=6, alpha=0.5)

    for plotidx00 in range(len(np.where(~np.isnan(aifl[Ch_next][fishN0[0]]))[1])):
        plt.plot(time[Ch_next][identity[Ch_next] == aifl[Ch_next][fishN0[0][0]][plotidx00]],
                 frequency[Ch_next][
                     identity[Ch_next] == aifl[Ch_next][fishN0[0][0]][plotidx00]],
                 'black', LineWidth=5, alpha=0.5)

    for plotidx1 in range(len(np.where(~np.isnan(aifl[Ch_next][fishN1[0]]))[1])):
        plt.plot(time[Ch_next][identity[Ch_next] == aifl[Ch_next][fishN1[0][0]][plotidx1]],
                 frequency[Ch_next][
                     identity[Ch_next] == aifl[Ch_next][fishN1[0][0]][plotidx1]],
                 'green', LineWidth=4, alpha=0.5)

    for plotidx11 in range(len(np.where(~np.isnan(aifl[Ch][fishN1[0]]))[1])):
        plt.plot(time[Ch][identity[Ch] == aifl[Ch][fishN1[0][0]][plotidx11]],
                 frequency[Ch][identity[Ch] == aifl[Ch][fishN1[0][0]][plotidx11]],
                 'yellow', LineWidth=3, alpha=0.5)

    plt.plot(t0, f0, 'red', LineWidth=2)
    plt.plot(t1, f1, 'blue', LineWidth=1)
    plt.title('ch {}: fish {};   ch {}: fish {}'.format(Ch, fishN0[0][0], Ch_next, fishN1[0][0]))
    plt.show()


def plot_together(time, frequency, identity, aifl, fishN, farbe):
    """ plots all the frequency-time traces of all identities of one fish

    Parameters
    ----------
    time:       2-D array
                loaded time array; 1_D: channel;         2_D: arbitrary time points [int]
    frequency:  2-D array
                loaded frequency array; 1_D: channel;    2_D: frequencies [Hz]
    identity:   2-D array
                loaded ident array; 1_D: channel;        2_D: identities [int]
    aifl:       3-D array
                all identity fish list; 1_D: channel;    2_D: fish number;      3_D: fish identities
    fishN:      int
                number of the fish for aifl
    farbe:      str
                hex color for the plot

    Returns
    -------
    nothing
    """

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('fish {}'.format(fishN))
    for channel_idx in range(16):
        fish1 = aifl[channel_idx, fishN, ~np.isnan(aifl[channel_idx, fishN])]
        r1 = len(fish1)
        print(fish1)
        for len_idx1 in range(r1):
            plt.plot(time[channel_idx][identity[channel_idx] == fish1[len_idx1]],
                     frequency[channel_idx][identity[channel_idx] == fish1[len_idx1]],
                     Linewidth=1, label=str(channel_idx) + ',' + str(fish1[len_idx1]), color=farbe)
    # plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    legend_without_duplicate_labels(fig, ax)
    plt.show()

    return ax


def plot_all_channels(time, frequency, identity, aifl, fishN1, fishN2=None):
    """ plots all the traces of each channel in a different subfigures

    Parameters
    ----------
    time:       2-D array
                loaded time array; 1_D: channel;         2_D: arbitrary time points [int]
    frequency:  2-D array
                loaded frequency array; 1_D: channel;    2_D: frequencies [Hz]
    identity:   2-D array
                loaded ident array; 1_D: channel;        2_D: identities [int]
    aifl:       3-D array
                all identity fish list; 1_D: channel;    2_D: fish number;      3_D: fish identities
    fishN1:     int
                fish number of the first fish
    fishN2:     int; optional
                fish number of the second fish

    Returns
    -------
    nothing
    """

    xcounter = 0
    ycounter = 0

    fig1, axs1 = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(16, 14))
    fig1.suptitle('fish {};  fish {}'.format(fishN1, fishN2))
    for channel_idx in range(16):
        fish1 = aifl[channel_idx, fishN1, ~np.isnan(aifl[channel_idx, fishN1])]
        r1 = len(fish1)
        print(fish1)
        for len_idx1 in range(r1):
            axs1[ycounter, xcounter].plot(time[channel_idx][identity[channel_idx] == fish1[len_idx1]],
                                          frequency[channel_idx][identity[channel_idx] == fish1[len_idx1]],
                                          'gray', Linewidth=3)
        # if fishN2 is given
        if fishN2 is not None:
            fish2 = aifl[channel_idx, fishN2, ~np.isnan(aifl[channel_idx, fishN2])]
            r2 = len(fish2)
            for len_idx2 in range(r2):
                axs1[ycounter, xcounter].plot(time[channel_idx][identity[channel_idx] == fish2[len_idx2]],
                                              frequency[channel_idx][identity[channel_idx] == fish2[len_idx2]],
                                              'blue', Linewidth=1)
        if xcounter == 3:
            xcounter = 0
            ycounter = ycounter + 1
        else:
            xcounter = xcounter + 1
    plt.show()


def legend_without_duplicate_labels(fig, ax):
    """ creats a legend without duplicated labels

    Parameters
    ----------
    fig: figure handle
    ax: ax handle

    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for lwdl_idx, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:lwdl_idx]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1, 0.4), loc="center right", bbox_transform=fig.transFigure, ncol=1)


def running_3binsizes(ipp, sampling_rate):
    """

    Parameters
    ----------
    ipp:            2-D array
                    interpolated power pancake; power of the signal over time for each channel
                    1_D: channel;       2_D: power of the signal over time
    sampling_rate:  int
                    sampling rate of the recordings

    Returns
    -------
    run_mean:   2-D array
                running mean with different bin sizes,
                first dimension: mean, second: bin size (from the running function)
    run_std:    2-D array
                running std with different bin sizes,
                first dimension: std, second: bin size (from the running function, same bin size as mean)

    """

    # lists
    running_mean = []
    running_std = []

    # max power in which channel of ipp
    max_ch = np.argmax(ipp, axis=1)

    # bin sizes
    bin1 = int(np.floor(15 * 60 / sampling_rate))
    bin2 = int(np.floor(30 * 60 / sampling_rate))
    bin3 = int(np.floor(60 * 60 / sampling_rate))

    # steps with steps 1/2 of bin size
    steps1 = np.arange(0, int(len(max_ch) - bin1), 7.5 * 60)
    steps2 = np.arange(0, int(len(max_ch) - bin2), 15 * 60)
    steps3 = np.arange(0, int(len(max_ch) - bin2), 30 * 60)

    # make running mean, std and sem
    for bins, step in zip([bin1, bin2, bin3], [steps1, steps2, steps3]):
        bin_mean = np.full([len(max_ch)], np.nan)
        bin_std = np.full([len(max_ch)], np.nan)

        # for i in range(int(len(max_ch) - bins)):
        for i in step:
            i = int(i)
            bin_mean[int(i + np.floor(bins / 2))] = np.mean(max_ch[i:bins + i])
            bin_std[int(i + np.floor(bins / 2))] = np.std(max_ch[i:bins + i])

        running_mean.append(bin_mean)
        running_std.append(bin_std)

    return running_mean, running_std


def running_variablebinsize(ipp, sampling_rate):
    ''' calculates the running mean and running std of the interpolated power pancake
    with two different bin sizes and a step size half of the bin size

    Parameters
    ----------
    ipp:            2-D array
                    interpolated power pancake; power of the signal over time for each channel
                    1_D: channel;       2_D: power of the signal over time
    sampling_rate:  int
                    sampling rate of the recordings

    Returns
    -------
    running_mean:   2-D array
                    running mean over the time axis with two different bin sizes and step sizes, which are also
                    dependent on the length of the time trace;
                    1_D: different bin size [2];    2_D: MEAN, (with steps half of the bin size)
    running_std:    2-D array
                    running std over the time axis with two different bin sizes and step sizes, which are also
                    dependent on the length of the time trace;
                    1_D: different bin size [2];    2_D: STD, (with steps half of the bin size)

    '''

    # lists
    running_mean = []
    running_std = []

    # max power in which channel of ipp
    max_ch = np.argmax(ipp, axis=1)

    # calculate bin sizes
    if len(ipp) <= 1800 / sampling_rate:  # all time traces < 30 min
        bin1 = int(np.floor(60 / sampling_rate))
        bin2 = int(np.floor(2 * 60 / sampling_rate))
        steps1 = np.arange(0, int(len(max_ch) - bin1), 0.5 * 60)
        steps2 = np.arange(0, int(len(max_ch) - bin2), 1 * 60)

    elif len(ipp) <= 14400 / sampling_rate:  # all time traces < 4 h
        bin1 = int(np.floor(10 * 60 / sampling_rate))
        bin2 = int(np.floor(20 * 60 / sampling_rate))
        steps1 = np.arange(0, int(len(max_ch) - bin1), 5 * 60)
        steps2 = np.arange(0, int(len(max_ch) - bin2), 10 * 60)

    elif len(ipp) > 14400 / sampling_rate:  # all time traces > 4 h
        bin1 = int(np.floor(60 * 60 / sampling_rate))
        bin2 = int(np.floor(180 * 60 / sampling_rate))
        steps1 = np.arange(0, int(len(max_ch) - bin1), 30 * 60)
        steps2 = np.arange(0, int(len(max_ch) - bin2), 90 * 60)

    # make running mean, std and sem
    for bins, step in zip([bin1, bin2], [steps1, steps2]):
        bin_mean = np.full([len(max_ch)], np.nan)
        bin_std = np.full([len(max_ch)], np.nan)

        # for i in range(int(len(max_ch) - bins)):
        for i in step:
            i = int(i)
            bin_mean[int(i + np.floor(bins / 2))] = np.mean(max_ch[i:bins + i])
            bin_std[int(i + np.floor(bins / 2))] = np.std(max_ch[i:bins + i])

        running_mean.append(bin_mean)
        running_std.append(bin_std)
        running_mean = np.array(running_mean)
        running_std = np.array(running_std)

    return running_mean, running_std


def plot_running(x, run_mean, run_std, threshold, fish, farbe, fs, fs_ticks, lw):
    """

    Parameters
    ----------
    x:  1-D array
        x axis points in date time format
    run_mean:   2-D array
                running mean with different bin sizes,
                first dimension: mean, second: bin size (from the running function)
    run_std:    2-D array
                running std with different bin sizes,
                first dimension: std, second: bin size (from the running function, same bin size as mean)
    threshold:  1-D array
                threshold for the activity phases dependent on the run mean,
                length of array same as how many bin sizes where calculated
    fish:       int
                title for the figure
    farbe:      1-D list
                list with all available colors from the matplotlib.colors in hex-color code
    fs:         int;    fontsize
    fs_ticks:   int;    fontsize of ticks
    lw:         int;    line width

    Returns
    -------

    """

    fig = plt.figure(figsize=(16, 14))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[1, 0])
    ax4 = fig.add_subplot(spec[1, 1])

    plt.suptitle('fish {}'.format(fish), fontsize=fs + 2)
    ax_cntr = 0

    for ax in [ax1, ax2, ax3]:
        non_nan = np.arange(len(run_mean[ax_cntr]))[~np.isnan(run_mean[ax_cntr])]

        ax.plot(x[fish][non_nan], run_mean[ax_cntr][non_nan], '.', color=farbe[ax_cntr + 4])
        ax.fill_between(x[fish][non_nan], run_mean[ax_cntr][non_nan] + run_std[ax_cntr][non_nan],
                        run_mean[ax_cntr][non_nan] - run_std[ax_cntr][non_nan],
                        facecolor=farbe[ax_cntr + 4], alpha=0.5)

        ax4.plot(x[fish][non_nan], run_std[ax_cntr][non_nan], color=farbe[ax_cntr + 4],
                 label=threshold[ax_cntr])

        ax4.plot([x[fish][non_nan][0], x[fish][non_nan][-1]],
                 [threshold[ax_cntr], threshold[ax_cntr]],
                 color=farbe[ax_cntr + 4])
        ax_cntr += 1

        ax.set_ylim([0, 15])
        ax.invert_yaxis()

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(width=lw - 2)
        ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

    ax1.set_xlabel('time [h]', fontsize=fs)
    ax1.set_ylabel('channel', fontsize=fs)

    plt.legend()