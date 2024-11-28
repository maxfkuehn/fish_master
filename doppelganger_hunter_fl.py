import numpy as np
from IPython import embed
from tqdm import tqdm
import datetime as dt
import os



def duplicate_hunter_fl(fishlist):
# this function looks for duplicates of entrys in fish list, where multiple fish have same CH,Id pairs and deletes them.
# So no fish is acidanntly doubled.
    pop_list = []
    pop_fish_nr = []
    for fish_nr in tqdm(range(len(fishlist))):
        index_counter = fish_nr+1
        if index_counter <= len(fishlist):
            for fl in fishlist[fish_nr]:

                checked_fish_nr = 0
                for check_fl in fishlist[index_counter:-1]:

                    real_fish_nr = checked_fish_nr+index_counter
                    if fl in check_fl:
                        pop_list.extend([fl])
                        pop_fish_nr.extend([real_fish_nr])

                    checked_fish_nr += 1

    for c in range(len(pop_list)):
        if pop_list[c] in fishlist[pop_fish_nr[c]]:
            fishlist[pop_fish_nr[c]].remove(pop_list[c])

    pop_emtpy = []

    for idx in range(len(fishlist)):
        if any(fishlist[idx]):
            continue
        else:
            pop_emtpy.extend([idx])

    pop_emtpy.reverse()

    global new_fish_list
    new_fish_list = fishlist.tolist()

    for pop in pop_emtpy:
        new_fish_list.pop(pop)

    return new_fish_list

def duplicate_hunter_fish(fish_list):
# this function looks for duplicates of entrys in within a fish in the list, where multiple pairs
# within one the fish might have same CH, Id pairs. Makes the Unique pairs.
# So no fish id pair is accidentally doubled.
    for fish_nr in tqdm(range(len(fish_list))):
        new_fish_list = list(set(map(tuple, fish_list[fish_nr])))
        fish_list[fish_nr] = [[int(pair[0]), int(pair[1])] for pair in new_fish_list]

    return fish_list

if __name__ == "__main__":
    # load fishlist
    # auswahl datensatz a=0 21.10.19 a=1 22.10.19
    a = 0  # select record day
    # auswahl raw b = 0 bearbeitet b = 1
    b = 1  # select raw or processed data

    # save path
    save_path = '/home/kuehn/Max_Masterarbeit/data/processed_raw_data'

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

    ##### import fish data:#######
    if b == 0:

        filename = sorted(os.listdir('../../../kuehn/data/analysed_data'))[a]
        fish_list = np.load('fishlist' + filename + '.npy', allow_pickle=True)
        filepath = ('../../../kuehn/data/analysed_data/' + filename)
    elif b == 1:
        filename = sorted(os.listdir(save_path))[a]
        fish_list = np.load(save_path + '/' + filename + '/fishlist.npy', allow_pickle=True)
        filepath = save_path + '/' + filename

    new_fish_list = duplicate_hunter_fl(fish_list)
    final_fish_list = duplicate_hunter_fish(new_fish_list)
    save_path2= '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'

    np.save(save_path2 + record_day + '/fishlist.npy', np.array(final_fish_list, dtype=object))
    np.save(save_path2 + record_day + '/old_fishlist.npy', np.array(fish_list, dtype=object))