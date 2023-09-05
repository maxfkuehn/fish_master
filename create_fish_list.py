import os
import numpy as np
from IPython import embed
from tqdm import tqdm

#load matching pairs
#auswahl datensatz a=0 22.10.19 a=1 23.10.19
a=1
filename = sorted(os.listdir('../../../kuehn/data/analysed_data'))[a]
mapa = np.load('mapa'+filename+'.npy')

#creating a list of fish with belonging list of channel and ids. Fish_list[0][0]
# -> first fish first channel and belonging id. Fish_list[0][0][:] -> all channels of first fish
# Fish_list[0][1][:] -> all ids of first fish

fish_list = []
fish_counter = len(fish_list)-1

for mp in tqdm(range(len(mapa))):
    ch_id_pair0 = [mapa[mp][0],mapa[mp][2]]
    ch_id_pair1 = [mapa[mp][1],mapa[mp][3]]

    if not fish_list:
        fish_list.append([ch_id_pair0])
        fish_list[0].append(ch_id_pair1)
        continue
    assigned = False
    for i in range(len(fish_list)):

        if ch_id_pair0 in fish_list[i] and ch_id_pair1 not in fish_list[i]:
            fish_list[i].append(ch_id_pair1)
            assigned = True
            break

        elif ch_id_pair0 not in fish_list[i] and ch_id_pair1 in fish_list[i]:
            fish_list[i].append(ch_id_pair0)
            assigned = True
            break

        elif ch_id_pair0 in fish_list[i] and ch_id_pair1  in fish_list[i]:
            assigned = True
            break

    if assigned == False:
        fish_list.append([ch_id_pair0])
        fish_list[-1].append(ch_id_pair1)




real_fish_counter = len(fish_list)

print(real_fish_counter)
print(fish_list)
#saving fishlist with data date
np.save('fishlist'+filename+'.npy', np.array(fish_list, dtype=object))



