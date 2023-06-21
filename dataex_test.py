import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import matplotlib.dates as mdates
import datetime as dt
from IPython import embed


class DataExtractor(object):
    # create a draggable rectangle on left click  selected to select data from plot to plot the grid postion
    # selected data in a subplot with a measurement grid from a recording. ? lets you delete selected data or throw
    # fish ids together
    def __init__(self, fish_list, x_grid, y_grid, start_date, filepath):
        # init variable are defined also plots are plottet in class
        # starting date and time of recodring
        self.start_date = start_date
        # fish data
        self.fish_list = fish_list
        self.ident = np.load(filepath + '/ident.npy', allow_pickle=True)
        self.sign = np.load(filepath + '/sign.npy', allow_pickle=True)
        self.freq = np.load(filepath + '/freq.npy', allow_pickle=True)
        self.timeidx = np.load(filepath + '/timeidx.npy', allow_pickle=True)
        self.times = np.load(filepath + '/times.npy', allow_pickle=True)
        ## grid data ###
        # electrode count

        # calculate all freq, time ident and channel arrays from fish_list

        self.af_dtime_fish = [] #list of time arrays of a fish for all fish
        self.af_freq_fish = [] # list of frequencies of a fish over all fish
        self.af_ident = [] # ident list
        self.af_ch_list = [] # chanel/electrode list for  fish over all fish

        for fish_nr in range(len(self.fish_list)):
            self.sf_freq = []
            self.sf_time = []
            self.sf_ident = []
            self.sf_ch = []
            for i in range(len(fish_list[fish_nr])):
                self.ch = self.fish_list[fish_nr][i][0]
                self.ids = self.fish_list[fish_nr][i][1]
                self.time_points = self.times[self.timeidx[self.ch][
                    self.ident[self.ch] == self.ids]]  # feändert für test for t in range(len(times)):
                # create a list with dt.datetimes for every idx of times of one fish:
                # for t in range(len(self.time_points)):
                # self.date_time = self.start_date + dt.timedelta(seconds=self.time_points[t])
                # self.sf_time.append(self.date_time)

                self.sf_time.append(self.time_points)
                self.give_every_index_ch = np.ones(len(self.time_points))*self.ch
                self.sf_ch.append(self.give_every_index_ch)

                self.fr = self.freq[self.ch][self.ident[self.ch] == self.ids]
                self.sf_freq.append(self.fr)

                self.sf_id = self.ident[self.ch][self.ident[self.ch] == self.ids]
                self.sf_ident.append(self.sf_id)
            # fish_time = np.concatenate(time_fish)
            self.sf_ident = np.concatenate(self.sf_ident)
            self.af_ident.append(self.sf_ident)

            self.sf_freq = np.concatenate(self.sf_freq)
            self.af_freq_fish.append(self.sf_freq)

            self.sf_time = np.concatenate(self.sf_time)
            self.af_dtime_fish.append(self.sf_time)

            self.sf_ch = np.concatenate(self.sf_ch)
            self.af_ch_list.append(self.sf_ch)

        # create subplot
        self.fig, [self.ax1, self.ax2] = plt.subplots(1, 2)

        # plot recorded frequency over time for every fish in subplot 1,1
        for f in range(len(self.af_dtime_fish)):
            self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o', picker=True,
                                                 pickradius=1)

            # self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S %d-%m-%Y'))

        # plot grid in subplot 1,2
        self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')
        self.indexX = None
        self.index = None
        self.indexY = None
        # define rectancle starting variables
        self.selectedX = []
        self.selectedY = []
        self.rect = patches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor='red')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax1.add_patch(self.rect)
        # define events
        self.ax1.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax1.figure.canvas.mpl_connect('motion_notify_event', self.on_motion_pressed)

        self.is_pressed = None

    def on_press(self, event):
        # on press get X0 and Y0
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.is_pressed = True
        self.ax2.cla()
        self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')
        self.ax1.figure.canvas.draw()

    def on_release(self, event):
        # on release get endpoints and caluclate rectangle and redraw plot with rectangle in it
        # select data in rectangle to plot it
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax1.figure.canvas.draw()

        self.is_pressed = False
        self.fish_id_counterX = []
        self.fish_id_counterY = []
        self.Xindex = []
        self.Yindex = []
        #####  select data in selection rectangle ########
        # unify starting position of rectangle selection, to alsways compare data from bottom left Xmin,Ymin to top right
        # Xmax,Ymax
        if self.y0 < self.y1:
            self.Ymin = self.y0
            self.Ymax = self.y1
        else:
            self.Ymin = self.y1
            self.Ymax = self.y0

        if self.x0 < self.x1:
            self.Xmin = self.x0
            self.Xmax = self.x1

        else:
            self.Xmin = self.x1
            self.Xmax = self.x0
        # find the data and index of points in the selection area in the plot/graph
        # find y values of data in selection range of y
        for f in range(len(self.af_freq_fish)):
            self.indexY = np.where((self.af_freq_fish[f] >= self.Ymin) & (self.af_freq_fish[f] <= self.Ymax))

            if len(self.indexY[0]) != 0:  # 1 because list in alist is 1 element
                self.fish_id_counterY.append(f)
                self.Yindex.append(self.indexY)

        # find x values of data within selection range of x and y
        for i in range(len(self.Yindex)):


            self.indexX = np.where((self.af_dtime_fish[self.fish_id_counterY[i]][self.Yindex[i]] >= self.Xmin) &
                                   (self.af_dtime_fish[self.fish_id_counterY[i]][self.Yindex[i]] <= self.Xmax))

            if len(self.indexX[0]) != 0:
                self.fish_id_counterX.append(self.fish_id_counterY[i])
                self.Xindex.append(self.indexX)
                print('it fucking worked', self.fish_id_counterX)

        self.index = self.Xindex
        self.selected_fish = self.fish_id_counterX

        ####### plotting selected data ############
        # plot selected grid i sublot 2
        self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')

        #get channel and plot
        for i in range(len(self.index)):
            self.ch_array = self.af_ch_list[self.selected_fish[i]][self.index[i][0]]
            self.unique_ch = set(self.ch_array)



        self.sel_data, = self.ax2.plot(self.ch_X, self.ch_Y, marker='o', linestyle='None',
                                       color='red')
        self.ax2.figure.canvas.draw()

    def on_motion_pressed(self, event):
        # also display rectagle in motion
        if self.is_pressed is True:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.ax1.figure.canvas.draw()


if __name__ == "__main__":
    ########## loading data ############
    # load fishlist
    # auswahl datensatz a=0 22.10.19 a=1 23.10.19
    a = 1  # select record day
    #### date time ######
    # definiere Zeitpunkte der Datei als Datum (datetime):
    start_date_0 = dt.datetime(2019, 10, 21, 13, 25, 00)
    start_date_1 = dt.datetime(2019, 10, 22, 8, 13, 00)

    if a == 0:
        start_date = start_date_0
    elif a == 1:
        start_date = start_date_1
    #### importiert electrode position grid ####
    from recording_grid_columbia_2019 import x_grid
    from recording_grid_columbia_2019 import y_grid

    ##### import fish data:#######
    filename = sorted(os.listdir('../../../kuehn/data/analysed_data'))[a]
    fish_list = np.load('fishlist' + filename + '.npy', allow_pickle=True)

    # load recorded data
    filepath = ('../../../kuehn/data/analysed_data/' + filename)

    dataextract = DataExtractor(fish_list, x_grid, y_grid, start_date, filepath)

    plt.show()
