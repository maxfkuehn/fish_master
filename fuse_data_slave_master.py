
import os
import numpy as np
from IPython import embed
from IPython import embed
###################################################################################################################
# load data
###################################################################################################################

# load data from master grid of one day
#auswahl datensatz a=0 22.10.19 a=1 23.10.19
a=1

filename_master = sorted(os.listdir('../../../kuehn/data/tracked_data/master'))[a]

print(filename_master)

ident_master = np.load('../../../kuehn/data/tracked_data/master/' + filename_master + '/all_ident_v.npy', allow_pickle=True)
freq_master = np.load('../../../kuehn/data/tracked_data/master/' + filename_master + '/all_fund_v.npy', allow_pickle=True)
timeidx_master = np.load('../../../kuehn/data/tracked_data/master/' + filename_master + '/all_idx_v.npy', allow_pickle=True)
spec_master = np.load('../../../kuehn/data/tracked_data/master/' + filename_master + '/all_spec.npy', allow_pickle=True)
sign_master = np.load('../../../kuehn/data/tracked_data/master/' + filename_master + '/all_sign_v.npy', allow_pickle=True)
times_master = np.load('../../../kuehn/data/tracked_data/master/' + filename_master + '/all_times.npy', allow_pickle=True)


# load data from slave grid of the same day as master grid
filename_slave = sorted(os.listdir('../../../kuehn/data/tracked_data/slave'))[a]

ident_slave = np.load('../../../kuehn/data/tracked_data/slave/' + filename_slave + '/all_ident_v.npy', allow_pickle=True)
freq_slave = np.load('../../../kuehn/data/tracked_data/slave/' + filename_slave + '/all_fund_v.npy', allow_pickle=True)
timeidx_slave = np.load('../../../kuehn/data/tracked_data/slave/' + filename_slave + '/all_idx_v.npy', allow_pickle=True)
spec_slave = np.load('../../../kuehn/data/tracked_data/slave/' + filename_slave + '/all_spec.npy', allow_pickle=True)
sign_slave = np.load('../../../kuehn/data/tracked_data/slave/' + filename_slave + '/all_sign_v.npy', allow_pickle=True)
times_slave = np.load('../../../kuehn/data/tracked_data/slave/' + filename_slave + '/all_times.npy', allow_pickle=True)

# fuse np arrays to make a single array list for all data of both grids of recorded from on the same day
ident = np.concatenate((ident_master, ident_slave)) # fuse data of slave and master grid for all  parameter
freq = np.concatenate((freq_master, freq_slave))
timeidx = np.concatenate((timeidx_master, timeidx_slave))
spec_list = np.concatenate((spec_master, spec_slave))
sign_list = np.concatenate((sign_master, sign_slave))
times = np.concatenate((times_master,times_slave))

dir_path = '../../../kuehn/data/analysed_data'
dir_day=dir_path+'/'+filename_master



if not os.path.exists(dir_day):           # check if Folder exist
    os.makedirs(dir_day)        # if not create new directory [current_path]


#save fused data with record date
np.save(dir_day + '/ident.npy', np.array(ident, dtype=object))
np.save(dir_day + '/freq.npy', np.array(freq, dtype=object)),
np.save(dir_day  + '/timeidx.npy', np.array(timeidx, dtype=object))
np.save(dir_day + '/sign.npy', np.array(sign_list, dtype=object))
np.save(dir_day + '/times.npy',np.array(times, dtype=object))

print('finished')
#################################################################################################################
