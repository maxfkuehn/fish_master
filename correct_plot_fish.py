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
#############################################################################################################
###### plotting of one fish sonogram and occurance of Channel-Id pairs in the recording grid cooridantes#####
#############################################################################################################

############### input parameters ########################
yes = {'y','Y'}
no = {'n','N'}
################## plotting #####################
# fish and time frame selection of plot window
fish= int(input('Choose fish to plot \n'))

#fish = 1 # fish number
time_frame = 15 # in min
###plotting parameter
#color array
color_array = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:pink']



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

#definiere Zeitpunkte der Datei als Datum (datetime):
start_date = dt.datetime(2019, 10, 21, 13, 25, 00 )
end_date = start_date + dt.timedelta(seconds = times[-1])
datetime_list = []
#create a list with with dt.datetimes for every idx of times:
for t in range(len(times)):
    new_time = start_date + dt.timedelta(seconds=times[t])
    datetime_list.append(new_time)


#array of elektrode channels sorted by position in grid related to x,y coordinates
m0 = np.arange(8,16)
m1 = np.arange(8)
s0 = np.arange(24,32)
s1 = np.arange(16,24)
ch_lists = m0[::-1],s0[::-1],m1[::-1], s1[::-1]
ch_pos_grid = np.concatenate(ch_lists,axis=0)


# xy coordinates of the grid
x_grid_2d = np.concatenate([np.zeros(16), np.ones(16)])
y_grid_2d = np.concatenate([np.arange(0,-16,-1), np.arange(0,-16,-1)])

#Abstand der Elektroden grid master rechten Seite
# Abstand zwischen Elektroden rechts ist aufegteilt in zwei Abstande, Mitte ist die position der Elektrode auf
# der Rechtenseite. Bsp. [55,55] -> elreode x-y haben Abtsand 110 und positionelektrode auf der anderen seite
# ist 55 von x und 55 y entfernt

Grid_master_right_data= [70, 30],[70, 35], [65, 45],[55, 50],[50, 50],[60, 45],[55, 55]
degree_master_left=[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
degree_master_right=[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

# calculate distance in master grid between elektrodes on the right row
end_diff_master_left =40

master_distance_left =[]

for i in range(len(Grid_master_right_data)-1):
    a=[Grid_master_right_data[i][1],Grid_master_right_data[i+1][0]]
    master_distance_left.append(a)
end_numbers = [Grid_master_right_data[-1][1]], end_diff_master_left
master_distance_left.append(end_numbers)
#Distancase slave grid, [0] dsitance and degree between slave and master

slave_distance_grid_data_left =[50,50],[50,50],[50,50],[65,35],[25,75],[25,75],[25,75],[25,75]
degree_slave_left = [30,30],[30,30],[30,30],[30,65],[65,65],[65,65],[65,65],[65,65]

slave_distance_grid_data_right = [70,35],[50,50],[50,50],[65,35],[75,25],[75,25],[75,25],[75,25]
degree_slave_right = [0,30], [30,30], [30,30], [30,65], [65,65], [65,65], [65,65],[65,65]

left_dist_list= [master_distance_left, slave_distance_grid_data_left]
left_dist_list=np.concatenate(left_dist_list)

left_degree_list = [degree_master_left, degree_slave_left]
left_degree_list = np.concatenate(left_degree_list)



right_dist_list=[Grid_master_right_data, slave_distance_grid_data_left]
right_dist_list=np.concatenate(right_dist_list)

right_degree_list =[degree_master_right, degree_slave_right]
right_degree_list = np.concatenate(right_degree_list)


y_coordinates_left = [0]
y_coordinates_right =[70]

x_coordinates_left = [0]
x_coordinates_right = [80]


for i in range(len(left_dist_list)):
    yd1 = np.cos(np.deg2rad(left_degree_list[i][0]))
    yl1 = np.int64(left_dist_list[i][0])
    a = yl1 *yd1
    yd2 = np.cos(np.deg2rad(left_degree_list[i][1]))
    yl2 = np.int64(left_dist_list[i][1])
    b = yl2 * yd2
    yl = y_coordinates_left[-1] - a -b
    y_coordinates_left.append(yl)

    yds1 = np.cos(np.deg2rad(left_degree_list[i][0]))
    yr1 = np.int64(left_dist_list[i][0])
    bb = yr1 *yd1
    yds2 = np.cos(np.deg2rad(left_degree_list[i][1]))
    yr2 = np.int64(left_dist_list[i][1])
    aa = yr2 *yds2
    yr = y_coordinates_right[-1] - aa - bb
    y_coordinates_right.append(yr)

for i in range(len(left_dist_list)):

    xd1 = np.sin(np.deg2rad(left_degree_list[i][0]))
    x1 = np.int64(left_dist_list[i][0])*xd1

    xd2 = np.sin(np.deg2rad(left_degree_list[i][1]))
    x2 = np.int64(left_dist_list[i][1])*xd2

    xl = x_coordinates_left[-1] +x1 +x2

    x_coordinates_left.append(xl)

    xl1 = np.sin(np.deg2rad(right_degree_list[i][0]))
    xr1 = np.int64(right_dist_list[i][0]) * xd1

    xl2 = np.sin(np.deg2rad(right_degree_list[i][1]))
    xr2 = np.int64(right_dist_list[i][1]) * xd2

    xr = x_coordinates_right[-1] + xr1 + xr2
    x_coordinates_right.append(xr)

x_coordinates= [x_coordinates_left, x_coordinates_right]
y_coordinates= [y_coordinates_left, y_coordinates_right]


x_grid = np.concatenate(x_coordinates)
y_grid = np.concatenate(y_coordinates)


######## overall data structure and parameters ###########
#load time occurances of all ids one fish and add to one list
fish_nr = fish - 1
time_fish = []
freq_fish = []

for i in range(len(fish_list[fish_nr])):
    ch = fish_list[fish_nr][i][0]
    ids = fish_list[fish_nr][i][1]
    t = times[timeidx[ch][ident[ch] == ids]]
    fr = freq[ch][ident[ch] == ids]

    time_fish.append(t)
    freq_fish.append(fr)

######### plot over whole time of one fish ( Test for later) ###########
fig = plt.figure()
all_id= fig.add_subplot((111))
for id_nr in range(len(fish_list[fish_nr])):
    date_fish =[]
    for lol in range(len(time_fish[id_nr])):
        new_time = start_date + dt.timedelta(seconds=time_fish[id_nr][lol])
        date_fish.append(new_time)
    all_id.plot_date(date_fish,freq_fish[id_nr],'-')
myFmt = mdates.DateFormatter('%d %m %Y  %H:%M:%S')
all_id.xaxis.set_major_formatter(myFmt)
plt.setp(all_id.get_xticklabels(), ha='right', rotation=45, rotation_mode='anchor')
plt.show(block=False)
#all_id.xaxis.set_major_formatter(mdates.DateFormatter('%Hh:%M:%S')) #setzt achse auf tageszeit
################################## plot in x min time frames ############################################
# getting start and end time where fish appeared first
start_time = min(np.concatenate(time_fish))
end_time = max(np.concatenate(time_fish))

#ploting
window_time = time_frame*1000/60
out_of_end_time = False
not_empty = False
window_nr = 0

while out_of_end_time == False:

    window_nr += 1
    w_start = start_time + window_time*(window_nr-1)
    w_end = start_time + window_time*window_nr

    if w_start >= end_time:
        out_of_end_time = True
    else:
        fig, (sonagramm,coordinates) = plt.subplots(1, 2)
        #X Achse auf gemessene Tageszeit anpassen bei spectrogram
        sonagramm.xaxis.set_major_formatter(mdates.DateFormatter('%Hh:%M:%S')) #setzt achse auf tageszeit
        plt.setp(sonagramm.get_xticklabels(), ha='right', rotation=30, rotation_mode='anchor') #macht was am x label :D

        #coordinaten gelich groß setzen:
        coordinates.set_aspect('equal')


        plt.title(f'Fish {fish_nr}, window nr {window_nr},')
        plt.ion()


        sonagramm.set_xlim(w_start, w_end)
        #coordinates.set_xlim(-0.5, 1.5)
        coordinates.set_title('Place in Grid')
        sonagramm.set_title('Spectocram for 15 min time window')
        #full_plot.set_title('Complete Spectrogram')


        for nr_id in range(len(time_fish)):
            c = np.random.rand(3)

            idx =np.logical_and(time_fish[nr_id] >= w_start,time_fish[nr_id] <= w_end)
            #idx = np.where(logical_idx,time_fish[nr_id])
            if all(idx) == False:
                continue
            else:

                #sonagram
                time_array = time_fish[nr_id][idx]
                date_array = []
                for t in range(len(time_array)):
                    date_array.append(start_date + dt.timedelta(seconds=time_array[t]))
                freq_array = freq_fish[nr_id][idx]

                sonagramm.plot_date(date_array,freq_array, marker='.', color=c)
                check_part =sonagramm
                # part in full_plot
                #full_plot.plot(time_array, freq_array, marker='.', color=c)
                #coordinates plot

                x_pos = x_grid[ch_pos_grid == fish_list[fish_nr][nr_id][0]] + random()*10
                y_pos = y_grid[ch_pos_grid == fish_list[fish_nr][nr_id][0]] + random()*10

                coordinates.plot(x_grid, y_grid, 'k*')
                xy = coordinates.plot(x_pos, y_pos, 'o', color=c)
                plt.show(fig)

                # aks if id matches or not, if not del id from list
                choice= input('Does the new Id match ? (y/n) \n')

                if choice in yes:
                    average = sum( freq_array) / len( freq_array)
                    print("Id checked. Last matched Id had a average Freq of ",average, "Hz")

                    pass
                elif choice in no:
                    #entfernt falsche IDs aus dem Plot.
                    no_match = check_part.pop(0)
                    no_match_pos = xy.pop(0)
                    no_match.remove()
                    no_match_pos.remove()
                    plt.show()

                    #liste mit falsch zugewiesenen IDs die aus fisch gelöscht werden
                    print(' Id deleted ')


                    pass
                else:
                    sys.stdout.write("Please respond with 'y' or 'n' \n")
        # skipp empty plots
        plt.close()
















