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
import math
# load own functions
from fish_list_unpacker import fish_list_unpacker as flu



if __name__ == "__main__":

    # auswahl datensatz a=0 21.10.19 a=1 22.10.19
    a = 1  # select record day
    # data hist ready
    b = 0  # 0 no, 1 yes
    # save path
    load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'

    # histogram settings
    bin_width = 60  # in seconds

    if b == 0:
        # load fishlist
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
        #### importiert electrode position grid ####
        from recording_grid_columbia_2019 import x_grid
        from recording_grid_columbia_2019 import y_grid
        from recording_grid_columbia_2019 import electrode_id_list
    ##### import fish data:#######