import datetime as dt
import math
import os
import random as rn
import statistics as stc

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from IPython import embed
from tqdm import tqdm

# load own functions
from fish_list_unpacker import fish_list_unpacker as flu
#### importiert electrode position grid ####
from recording_grid_columbia_2019 import x_grid
from recording_grid_columbia_2019 import y_grid
from recording_grid_columbia_2019 import electrode_id_list
from recording_grid_columbia_2019 import electrode_arrangement_list

def grid_middle_line_calculator (x_grid_value, y_grid_value, electrode_id_list, electrode_arrangement_list):
    """
    Calculates the middle line of grid.

    Calculation of a middle gridline by calculating middle of opposing ELectrodes of each String.

    Parameters:
    - X values of every electrode.
    - y value of every electrode.
    - 

    Returns:
    - Middle line as list of x and y pairs [[x1,y1],...,[x-1,y-1]]
    
    Example:
    Electrode 1 and 7 are on opposing sides in real grid.
    First order Electrodes according to electrode arrangement list, so 2 and 7 are opposite then:
    Middle X = (X_electrode 1 + X_electrode 7)/2
    Same for Y Value of middle-point
    """

    #reorder x_grid and y_grid according to electrode_arrangement
    reorder_x = dict(zip(electrode_id_list,x_grid_value))
    reorder_y = dict(zip(electrode_id_list,y_grid_value))

    x_grid_sorted = [reorder_x[element] for element in electrode_arrangement_list]

    y_grid_sorted = [reorder_y[element] for element in electrode_arrangement_list]

    #calculates the middleline of the given grid
    middle_line_grid=[]
    for even in range(0, len(x_grid_sorted), 2):
        odd = even + 1
        middle_x = (x_grid_sorted[even] + x_grid_sorted[odd]) / 2
        middle_y = (y_grid_sorted[even] + y_grid_sorted[odd]) / 2
        middle_line_grid.append([middle_x, middle_y])
    #return
    return middle_line_grid

if __name__ == "__main__":

    midle_line = grid_middle_line_calculator(x_grid, y_grid, electrode_id_list, electrode_arrangement_list)
    mlg = midle_line
    x_coords = [point[0] for point in mlg]
    y_coords = [point[1] for point in mlg]

    # Plot the data points
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Points')
    plt.grid(True)
