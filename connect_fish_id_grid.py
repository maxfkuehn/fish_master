import matplotlib.pyplot as plt
import os
import numpy as np
from IPython import embed
from tqdm import tqdm


if __name__ == '__main__':


    #load parameters
    # auswahl datensatz a=0 22.10.19 a=1 23.10.19
    a = 1
    filename= sorted(os.listdir('../../../kuehn/data/analysed_data'))[a]
    filepath =('../../../kuehn/data/analysed_data/' + filename)

    #load fused grid data
    ident = np.load(filepath + '/ident.npy', allow_pickle=True) #identity data (number)
    sign = np.load(filepath + '/sign.npy', allow_pickle=True)  #
    freq = np.load(filepath + '/freq.npy', allow_pickle=True) #frequency data
    timeidx = np.load(filepath + '/timeidx.npy', allow_pickle=True) #time idx

    ##### check if for overlapping unique IDs and asignt them to a fish ####

    #get two list : 1: possible channels is the idx that indicates in which channel a unique id is recorded
    #               2: possible ids: unique ids  per channel as one array , channel idx by pos_channel
    pos_channel = []
    pos_id = []
    #parameters for time check
    tolerance = 60 # [s]
    total_tolerance = 2*tolerance
    match_counter = 0
    non_match_counter = 0
    #matched pair list
    mapa = []

    # find me for each channel the unique identities and save it as an array also save the channel idx
    for index in range(len(ident)):
            uni_ident = np.unique(ident[index][~np.isnan(ident[index])]). astype(int)
            pos_id = np.append(pos_id,uni_ident). astype(int)
            channel_idx = np.full((1,len(uni_ident)),index). astype(int)
            pos_channel = np.append(pos_channel,channel_idx). astype(int)

    #compare ids for time and frequency overlap to identify th same fish in all other channels and merge ids
    for i in tqdm(np.arange(len(pos_id))):
        ch0 = pos_channel[i]
        id0 = pos_id[i]
        # timeidx window of id0 appears in recording

        t0 = timeidx[ch0][(ident[ch0] == id0)]
        t0_start = t0[0]
        t0_end = t0[-1]

        #frequency

        #start index conecte channels: where ch0+1 starts to only compare id with other channels

        if ch0 == pos_channel[-1]:
            break

        else:
            freq0 = freq[ch0][(ident[ch0]) == id0]
            start_idx_c_ch = np.where(pos_channel > ch0)[0][0]
            #conmparing with every other channel
            for j in np.arange(start_idx_c_ch,len(pos_id)):
                ch1 = pos_channel[j]
                id1 = pos_id[j]

                #time idx id1

                t1 = timeidx[ch1][(ident[ch1] == id1)]
                t1_start = t1[0]
                t1_end = t1[-1]
                #freq id1
                freq1 = freq[ch1][(ident[ch1] == id1)]


                # time with tolerance window
                t0_start_tol = t0_start - tolerance
                t0_end_tol = t0_end + tolerance
                t1_start_tol = t1_start - tolerance
                t1_end_tol = t1_end + tolerance

                #looking for time match of ids (overlap with tolerance)
                match = False



                if (t0_start_tol <= t1_start_tol) and (t1_start_tol <= t0_end_tol) and (t0_end_tol <= t1_end_tol):
                    # beispiel overlap: to  start o-------o end
                    #                   t1   start    o------o end
                    match = True
                elif (t1_start_tol <= t0_start_tol) and (t0_start_tol <= t1_end_tol) and (t1_end_tol <= t0_end_tol):
                    # beispiel overlap: to          start o-------o end
                    #                   t1   start   o------o end
                    match = True
                elif (t0_start_tol <= t1_start_tol) and (t0_end_tol >= t1_end_tol):
                    # beispiel overlap: to    start o-------o end
                    #                   t1      start o---o end

                    match = True
                elif (t1_start_tol <= t0_start_tol) and (t1_end_tol >= t0_end_tol):
                    # beispiel overlap: to    start o---o end
                    #                   t1  start o-------o end
                    match = True
                else:
                    non_match_counter += 1
                    pass



                #vergleichen der median frequenz , falls diff unter 2 match und als paar id und channel gespeichert
                window_times = sorted(np.array([t0_start_tol, t0_end_tol, t1_start_tol, t1_end_tol]))[1:3]
                if match:
                    match_counter += 1
                    f0_box_min = np.median(freq0[np.isclose(t0, window_times[0], atol=total_tolerance)])
                    f0_box_max = np.median(freq0[np.isclose(t0, window_times[1], atol=total_tolerance)])
                    f1_box_min = np.median(freq1[np.isclose(t1, window_times[0], atol=total_tolerance)])
                    f1_box_max = np.median(freq1[np.isclose(t1, window_times[1], atol=total_tolerance)])
                    pair= [ch0, ch1, id0, id1]
                    if f0_box_min.size > 0 and f1_box_min.size > 0:
                        fdiff0 = abs(f0_box_min - f1_box_min)
                        if fdiff0 <= 2: #[Hz]
                            mapa.append(pair.copy())

                        else:
                            continue
                    elif f0_box_max.size > 0 and f1_box_max.size > 0:
                        fdiff1 = abs(f0_box_max - f1_box_max)
                        if fdiff1 <= 2:
                            mapa.append(pair.copy())

                        else:
                            continue

    np.save('mapa'+filename+'.npy', mapa)
    print('finished')