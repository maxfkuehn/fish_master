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
from doppelganger_hunter_fl import duplicate_hunter_fl
import copy

# Data extracter
class DataExtractor(object):
    # create a draggable rectangle on left click  selected to select data from plot to plot the grid postion
    # selected data in a subplot with a measurement grid from a recording. ? lets you delete selected data or throw
    # fish ids together
    def __init__(self, fish_list, x_grid, y_grid, electrode_id_list, start_date, filepath):
        # init variable are defined also plots are plottet in class
        # starting date and time of recodring
        self.start_date = start_date
        # fish data
        self.fish_list = fish_list
        self.old_fish_list = self.fish_list
        self.ident = np.load(filepath + '/ident.npy', allow_pickle=True)
        self.sign = np.load(filepath + '/sign.npy', allow_pickle=True)
        self.freq = np.load(filepath + '/freq.npy', allow_pickle=True)
        self.timeidx = np.load(filepath + '/timeidx.npy', allow_pickle=True)
        self.times = np.load(filepath + '/times.npy', allow_pickle=True)
        ## grid data ###
        # electrode grid
        self.y_grid = np.array(y_grid)
        self.x_grid = np.array(x_grid)
        self.electrode_id_list = np.array(electrode_id_list)
        self.electrode_pos = electrode_id_list
        # calculate all freq, time ident and channel arrays from fish_list

        self.af_dtime_fish = [] #list of time arrays of a fish for all fish
        self.af_freq_fish = [] # list of frequencies of a fish over all fish
        self.af_ident = [] # ident list
        self.af_ch_list = [] # chanel/electrode list for  fish over all fish
        self.af_timeidx = [] # time index for all fish
        self.af_sign = []

        for fish_nr in range(len(self.fish_list)):
            self.sf_freq = []
            self.sf_time = []
            self.sf_ident = []
            self.sf_ch = []
            self.sf_tidx = []
            self.sf_sign = []

            for i in range(len(self.fish_list[fish_nr])):
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

                self.st_time_idx= self.timeidx[self.ch][self.ident[self.ch] == self.ids]
                self.sf_tidx.extend(self.st_time_idx)

                self.st_sign_idx = self.sign[self.ch][self.ident[self.ch] == self.ids]
                self.sf_sign.extend(self.st_sign_idx)

            # fish_time = np.concatenate(time_fish)
            self.sf_ident = np.concatenate(self.sf_ident)
            self.af_ident.append(self.sf_ident)

            self.sf_freq = np.concatenate(self.sf_freq)
            self.af_freq_fish.append(self.sf_freq)

            self.sf_time = np.concatenate(self.sf_time)
            self.af_dtime_fish.append(self.sf_time)

            self.sf_ch = np.concatenate(self.sf_ch)
            self.af_ch_list.append(self.sf_ch)

            self.af_timeidx.append(self.sf_tidx)

            self.af_sign.append(np.concatenate(self.sf_sign))


        # highest id counter
        self.highest_id = []

        for i in range(len(self.af_ident)):

            self.max_id_ch = max(self.af_ident[i])
            self.highest_id.append(self.max_id_ch)

        self.highest_id = max(self.highest_id)

        # create subplot
        self.fig, [self.ax1, self.ax2] = plt.subplots(1, 2)

        # plot recorded frequency over time for every fish in subplot 1,1
        for f in range(len(self.af_dtime_fish)):
            self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o', picker=True,
                                                 pickradius=5)
        self.ax1.set_ylabel('Freuqency [Hz]')
        self.ax1.set_xlabel('Time [s]')
        self.ax2.set_ylabel('Distance [cm]')
        self.ax2.set_xlabel('Distance [cm]')
            # self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S %d-%m-%Y'))
        ####
        # plot grid in subplot 1,2
        self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')
        self.indexX = None
        self.index = None
        self.indexY = None
        ####
        # define rectancle starting variables

        self.rect = patches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor='red')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax1.add_patch(self.rect)
        #######
        # rectangle subplot

        self.rect_sub = patches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor='red')
        self.x0_sub = None
        self.y0_sub = None
        self.x1_sub = None
        self.y1_sub = None
        self.ax2.add_patch(self.rect_sub)

        #define events

        self.ax1.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax2.figure.canvas.mpl_connect('button_press_event', self.grid_press)

        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion_pressed)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        #define marker/checkpoint variables
        self.grid_plot = None
        self.is_pressed = None
        self.make_rectangle = None
        self.there_is_a_box = None
        self.adjust_plot = None
        self.fish_fusion = None
        self.fish_divide = None
        self.selection = None
        self.undo = None
        self.select_grid = None
        self.is_subplot = None
        self.confirm = None
        self.selection_counter = 0
        self.function_blocker = None
        self.l_activator = None
        print('###################### \nPress E to plot the gridposition,\nPress T to put fish together \nPress Z to divide a fish at a selected location.\nPress O for Undo last Action\n######################')

########################################################################################
######################## update fish list function #####################################
    def update_fish_list(self):
        ### recalculate data from changed fish_list
        self.af_dtime_fish = []  # list of time arrays of a fish for all fish
        self.af_freq_fish = []  # list of frequencies of a fish over all fish
        self.af_ident = []  # ident list
        self.af_ch_list = []  # chanel/electrode list for  fish over all fish
        self.af_timeidx = []  # time index for all fish
        self.af_sign = []

        for fish_nr in range(len(self.fish_list)):
            self.sf_freq = []
            self.sf_time = []
            self.sf_ident = []
            self.sf_ch = []
            self.sf_tidx = []
            self.sf_sign = []



            for i in range(len(self.fish_list[fish_nr])):
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

                self.st_time_idx= self.timeidx[self.ch][self.ident[self.ch] == self.ids]
                self.sf_tidx.extend(self.st_time_idx)

                self.st_sign_idx = self.sign[self.ch][self.ident[self.ch] == self.ids]
                self.sf_sign.extend(self.st_sign_idx)


            # fish_time = np.concatenate(time_fish)

            if len(self.sf_ident[0]) > 1:
                self.sf_ident = np.concatenate(self.sf_ident)
            self.af_ident.append(self.sf_ident)

            self.sf_freq = np.concatenate(self.sf_freq)
            self.af_freq_fish.append(self.sf_freq)

            self.sf_time = np.concatenate(self.sf_time)
            self.af_dtime_fish.append(self.sf_time)

            self.sf_ch = np.concatenate(self.sf_ch)
            self.af_ch_list.append(self.sf_ch)

            self.af_timeidx.append(self.sf_tidx)

            self.sf_sign = np.concatenate(self.sf_sign)
            self.af_sign.append(self.sf_sign)



        self.ax1.cla()
        self.ax2.cla()

     # plot recorded frequency over time for every fish in subplot 1,1
        for f in range(len(self.af_dtime_fish)):
            self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',picker=True,pickradius=5)

        self.rect = patches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor='red')
        self.ax1.add_patch(self.rect)
        #######
        # rectangle subplot

        self.rect_sub = patches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor='red')
        self.ax2.add_patch(self.rect_sub)

        self.ax2.add_patch(self.rect_sub)
        self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')

        self.ax2.set_ylabel('Distance [cm]')
        self.ax2.set_xlabel('Distance [cm]')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        print('###################### \nPress E to plot the gridposition,\nPress T to put fish together \nPress Z to divide a fish at a selected location.\nPress O for Undo last Action\n######################')
    ###################################################################################################
        # save funcion
    def save_function(self):

        # save ident
        np.save(os.path.join((save_path + record_day),'ident.npy'),np.array(self.ident, dtype=object))
        # save sign
        np.save(os.path.join((save_path + record_day),'sign.npy'),np.array(self.sign, dtype=object))
        # save freq
        np.save(os.path.join((save_path + record_day),'freq.npy'),np.array(self.freq, dtype=object))
        # save timeidx
        np.save(os.path.join((save_path + record_day),'timeidx.npy'),np.array(self.timeidx, dtype=object))
        # save times
        np.save(os.path.join((save_path + record_day),'times.npy'),np.array(self.times, dtype=object))
        # save fish_list
        np.save(os.path.join((save_path + record_day), 'fishlist.npy'),np.array(self.fish_list, dtype=object))

        print('\nData changes are saved, revert by pressing O button\n')
################################################################################
##################### On Press ################################################
    ## press main plot
    def on_press(self, event):
        # on press get X0 and Y0 for E T ans Z button function
        if self.make_rectangle is True or self.fish_fusion is True:
            self.x0 = event.xdata
            self.y0 = event.ydata
            self.is_pressed = True
            self.ax2.cla()
            self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')
            self.ax2.set_ylabel('Distance [cm]')
            self.ax2.set_xlabel('Distance [cm]')
            plt.draw()


        if self.selection == 'click_z':
            self.x_z = event.xdata
            self.y_z = event.xdata
            self.point_diff_x = []
            self.point_diff_y = []

            for x in self.af_dtime_fish[self.sel_fish]:
                self.dif_x = abs(x-self.x_z)
                self.point_diff_x.append(self.dif_x)
            self.point_diff = np.array(self.point_diff_x)
            self.indx_closest_point_X = np.where(self.point_diff_x == min(self.point_diff_x))[0]

            self.frq_af = np.array(self.af_freq_fish)
            for y in self.frq_af[self.sel_fish][self.indx_closest_point_X]:
                self.dif_y = abs(y-self.y_z)
                self.point_diff_y.append(self.dif_y)

            self.point_diff = np.where(min(self.point_diff_y))[0][0]

            self.idx_z = self.indx_closest_point_X[self.point_diff]

            self.point_x = self.af_dtime_fish[self.sel_fish][self.idx_z]
            self.point_y = self.af_freq_fish[self.sel_fish][self.idx_z]

            self.sel_dot, = self.ax1.plot(self.point_x, self.point_y, marker='o',markeredgewidth=2, color='black')

            self.ax1.figure.canvas.draw()

            self.selection = 'confirm_z'

            print('Point selected.\nConfirm with Y or cancel with C\n')
    ## press on subplot
    def grid_press(self, event):
        if self.select_grid is True:
            self.x0_sub = event.xdata
            self.y0_sub = event.ydata
            self.is_pressed = True
            self.ax2.figure.canvas.draw()

##################################################################################################
############################ on release ##########################################################

    def on_release(self, event):
        #### Subplot events

        #rectangle in grid sub plot
        if self.select_grid is True:

            self.x1 = event.xdata
            self.y1 = event.ydata
            self.rect_sub.set_width(self.x1_sub - self.x0_sub)
            self.rect_sub.set_height(self.y1_sub - self.y0_sub)
            self.rect_sub.set_xy((self.x0_sub, self.y0_sub))
            self.ax2.set_ylabel('Distance [cm]')
            self.ax2.set_xlabel('Distance [cm]')
            self.ax2.figure.canvas.draw()

            self.is_pressed = False
            self.select_grid = False

            if self.y0_sub < self.y1_sub:
                self.Ymin_sub = self.y0_sub
                self.Ymax_sub = self.y1_sub
            else:
                self.Ymin_sub = self.y1_sub
                self.Ymax_sub = self.y0_sub

            if self.x0_sub < self.x1_sub:
                self.Xmin_sub = self.x0_sub
                self.Xmax_sub = self.x1_sub

            else:
                self.Xmin_sub = self.x1_sub
                self.Xmax_sub = self.x0_sub

            #  check for selected Grid ch
            self.lower_bar_y  = self.y_grid >= self.Ymin_sub
            self.upper_bar_y = self.y_grid <= self.Ymax_sub


            self.ygrid_idx = np.where(self.lower_bar_y & self.upper_bar_y)[0]

            self.lower_bar_x = self.x_grid[self.ygrid_idx] >= self.Xmin_sub
            self.upper_bar_x = self.x_grid[self.ygrid_idx] <= self.Xmax_sub

            self.xgrid_idx = np.where(self.lower_bar_x & self.upper_bar_x)

            self.grid_idx = self.ygrid_idx[self.xgrid_idx]

            self.sel_ch_grid = self.electrode_id_list[self.grid_idx] - 1

            self.new_sel_fish = []
            self.new_index = []
            self.new_sel_ch_list = []
            self.ch_appearance = []

            for fx in range(len(self.selected_fish)):

                self.fish_ch_array = self.af_ch_list[self.last_sel_fish[fx]][self.last_index[fx]]

                self.check_idx = pd.Series(self.fish_ch_array)
                self.idx_idx = np.asarray(np.where(self.check_idx.isin(self.sel_ch_grid))[0])


                if self.idx_idx.size > 0:

                    self.new_idx = self.last_index[fx][self.idx_idx]
                    self.new_sel_fish.extend([self.last_sel_fish[fx]])# append grid selected fish nr to a new list
                    self.new_index.append(self.new_idx)# append new index of selected data points to a new list

                    # get channel that are only part of the selected channels in grid:
                    self.fish_sel_ch = self.fish_ch_array[self.idx_idx]
                    self.new_sel_ch_list.extend(self.fish_sel_ch)# append ch of selected data points
                    self.ch_appearance.append(set(self.fish_sel_ch))
            self.new_unique_ch = set(self.new_sel_ch_list)
            self.new_pos_indx = []
            ## plot selected data in subplot
            for h in range(len(self.ch_appearance)):

                self.random_number = self.last_random_number[h]

                self.x_grid_sel = []
                self.y_grid_sel = []
                for x in self.ch_appearance[h]:

                    self.new_pos_indx.extend(np.where(x+1 == self.electrode_pos)[0])

                    for pos_idx in self.new_pos_indx:

                        self.x_grid_sel.extend([x_grid[pos_idx]+self.random_number])
                        self.y_grid_sel.extend([y_grid[pos_idx]+self.random_number])

                self.ax2.plot(self.x_grid_sel, self.y_grid_sel, marker='o', linestyle='None', markeredgewidth=2.2,
                          markerfacecolor='none', mec = 'r')
                self.ax2.set_ylabel('Distance [cm]')
                self.ax2.set_xlabel('Distance [cm]')

            ## plot selected data in main plot

            self.new_sel_x = []
            self.new_sel_y = []

            for r in range(len(self.new_sel_fish)):
                fish = self.new_sel_fish[r]
                for new_index in self.new_index[r]:
                    self.new_sel_x.extend([self.af_dtime_fish[fish][new_index]])
                    self.new_sel_y.extend([self.af_freq_fish[fish][new_index]])

            self.ax1.plot(self.new_sel_x,self.new_sel_y,marker='o', markerfacecolor='none',
                                                 markeredgecolor='yellow',markeredgewidth=0.5)

            self.ax2.figure.canvas.draw()
            self.ax1.figure.canvas.draw()
            print('Confirm Selection of Points to fuse them by pressing Y')
            self.confirm = True
        ##################################
        #Main Plot events
        if self.make_rectangle is True:
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
            self.index = []
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
                    self.X_index = np.array(self.indexX[0])
                    self.Xindex = self.Yindex[i][0][self.X_index]
                    self.index.append(self.Xindex)

            print('Identified fish', self.fish_id_counterX)
            # use index of Xindex in Yindex to find general index where everythign is true

            if len(self.fish_id_counterX) != 0:


                self.selected_fish = self.fish_id_counterX


            else:
                self.make_rectangle = None
                self.grid_plot = None
                self.function_blocker = None

        #####################################################
            #Funcion of E button
            #Plot selected data into recording grid subplot  with the same color
        #####################################################
            if self.grid_plot is True and self.function_blocker == 1:

                self.fish_color = [] # array with color of mainplot of a fish for subplot

                for c in self.selected_fish:
                    self.fish_color.append(self.ax1.get_lines()[c].get_color())

                ####### plotting selected data ############
                # plot selected grid i sublot 2
                self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')
                #save random generated number to be able to recall
                self.last_random_number = []
                self.selected_fish_sign = []
                #get channel and plot
                for i in range(len(self.index)):
                    self.ch_array = self.af_ch_list[self.selected_fish[i]][self.index[i]]
                    self.selected_fish_sign.extend([self.selected_fish[i]])
                    self.unique_ch = set(self.ch_array)
                    self.pos_indx= []
                    self.random_number = rn.randrange(-10, 11)
                    self.last_random_number.extend([self.random_number])
                    for x in self.unique_ch:

                        self.pos_indx.extend(np.where(x+1 == self.electrode_pos)[0])

                        self.y_grid_sel = []
                        self.x_grid_sel = []

                        for pos_idx in self.pos_indx:
                            self.x_grid_sel.extend([x_grid[pos_idx]+self.random_number])
                            self.y_grid_sel.extend([y_grid[pos_idx]+self.random_number])




                    self.ax2.plot(self.x_grid_sel,self.y_grid_sel, marker='o', linestyle='None', markeredgewidth=2, color=self.fish_color[i])
                    self.ax2.set_ylabel('Distance [cm]')
                    self.ax2.set_xlabel('Distance [cm]')

                self.last_sel_fish = self.selected_fish
                self.last_index = self.index
                self.last_x_grid_sel = self.x_grid_sel
                self.last_y_grid_sel = self.y_grid_sel

                self.make_rectangle = None
                self.grid_plot = None
                self.is_subplot = True
                self.function_blocker = None
                self.ax2.figure.canvas.draw()
                self.l_activater = True

                self.window_size = (self.Xmax - self.Xmin)
                self.date_time_window = str(dt.timedelta(seconds = self.window_size))
                print('Window time is ', self.date_time_window,'\n')
                print('To select data points in Grid Plot to fuse them Press U.')
                print('Press l to get Mean and Stdev sign of each ch\n')
            #############################################################
            #Function of the T button
            #Highlight connectable fish in Main and Subplot as same color
            #############################################################
            if self.selection == 1 and self.function_blocker == 2:
                if len(self.selected_fish) == 1:
                    self.Yfish1 = self.af_freq_fish[self.selected_fish[0]]
                    self.Xfish1 = self.af_dtime_fish[self.selected_fish[0]]
                    self.fish_selection, = self.ax1.plot(self.Xfish1, self.Yfish1, marker='o', markerfacecolor='none',
                                                 markeredgecolor='yellow',markeredgewidth=0.5)
                    self.ax1.figure.canvas.draw()

                    self.fish1 = self.selected_fish[0]
                    self.make_rectangle = None
                    self.selection = 'pause'
                    print('\nFirst fish selected. \nPlease press T to select second fish.')
                if len(self.selected_fish) > 1:
                    self.Yfish1 = self.af_freq_fish[self.selected_fish[0]]
                    self.Xfish1 = self.af_dtime_fish[self.selected_fish[0]]
                    self.fish_selection, = self.ax1.plot(self.Xfish1, self.Yfish1, marker='o', markerfacecolor='none',
                                                         markeredgecolor='yellow', markeredgewidth=0.5)
                    self.ax1.figure.canvas.draw()

                    self.sel_fish_idx_range = len(self.selected_fish) - 1
                    self.fish1 = self.selected_fish[0]
                    self.make_rectangle = None
                    self.selection = 'multiple_selected_fish_t_one'
                    print('\nMultiple Fish selected.\nSelected Fish Nr:', self.fish1, '.\nPress T to confirm selection of fish and select point of split .\n Or use A and D keys to switch between different selected fish\n')
            if self.selection == 2:
                if len(self.selected_fish) == 1:
                    self.Yfish2 = self.af_freq_fish[self.selected_fish[0]]
                    self.Xfish2 = self.af_dtime_fish[self.selected_fish[0]]
                    self.fish_selection, = self.ax1.plot(self.Xfish2, self.Yfish2, marker='o', markerfacecolor='none',markeredgecolor='yellow',markeredgewidth=1)
                    self.ax1.figure.canvas.draw()
                    self.selection = 'confirm_t'
                    self.fish2 = self.selected_fish[0]
                    print('\nSlection of fish complete. \nPress Y to confirm fish fusion or C to cancel.')

                if len(self.selected_fish) > 1:
                    self.Yfish2 = self.af_freq_fish[self.selected_fish[0]]
                    self.Xfish2 = self.af_dtime_fish[self.selected_fish[0]]
                    self.fish_selection, = self.ax1.plot(self.Xfish2, self.Yfish2, marker='o', markerfacecolor='none',
                                                         markeredgecolor='yellow', markeredgewidth=0.5)
                    self.ax1.figure.canvas.draw()

                    self.sel_fish_idx_range = len(self.selected_fish) - 1
                    self.fish2 = self.selected_fish[0]
                    self.make_rectangle = None
                    self.selection = 'multiple_selected_fish_t_two'
                    print(
                    '\nMultiple Fish selected. Selected fish:',self.fish2,'.\nPress Y to confirm selection of fish '
                                                                          'and select point of split .\n Or use A and'
                                                                          ' D keys to switch between different '
                                                                          'selected fish\n')



            #####################################################
            # Funcion of Z button
            # Select a single fish, select a data point where u cut it into 2 fish
            #####################################################
            if self.selection == 'pressed_z' and self.function_blocker == 3:
                if len(self.selected_fish) == 1:

                    self.Yfish = self.af_freq_fish[self.selected_fish[0]]
                    self.Xfish = self.af_dtime_fish[self.selected_fish[0]]

                    self.fish_plot, = self.ax1.plot(self.Xfish, self.Yfish, marker='o', markerfacecolor='none',
                                                 markeredgecolor='yellow', markeredgewidth=0.5)
                    self.ax1.figure.canvas.draw()

                    self.sel_fish = self.selected_fish[0]
                    self.make_rectangle = None
                    self.selection = 'selected_z'
                    print('\nFish selected. \nPlease press Z again to select point of split by clicking onto it.\n')

                elif len(self.selected_fish) > 1:
                    self.Yfish = self.af_freq_fish[self.selected_fish[0]]
                    self.Xfish = self.af_dtime_fish[self.selected_fish[0]]

                    self.fish_selection, = self.ax1.plot(self.Xfish, self.Yfish, marker='o', markerfacecolor='none',
                                                         markeredgecolor='yellow', markeredgewidth=0.5)
                    self.ax1.figure.canvas.draw()

                    self.sel_fish_idx_range = len(self.selected_fish) - 1
                    self.sel_fish = self.selected_fish[0]
                    self.make_rectangle = None
                    self.selection = 'multiple_selected_fish'
                    print('\nMultiple Fish selected. \nPress Z to confirm selection of fish and select point of split .\n Or use A and D keys to switch between different selected fish\n')
                else:
                    self.selection = None
####################################################################################################
######################## on motion #################################################################
    def on_motion_pressed(self, event):
        # also display rectagle in motion
        if self.is_pressed is True:
            if self.grid_plot is True:
                self.x1 = event.xdata
                self.y1 = event.ydata
                self.rect.set_width(self.x1 - self.x0)
                self.rect.set_height(self.y1 - self.y0)
                self.rect.set_xy((self.x0, self.y0))
                self.ax1.figure.canvas.draw()

            if self.select_grid is True:
                self.x1_sub = event.xdata
                self.y1_sub = event.ydata

                self.rect_sub.set_width(self.x1_sub - self.x0_sub)
                self.rect_sub.set_height(self.y1_sub - self.y0_sub)
                self.rect_sub.set_xy((self.x0_sub, self.y0_sub))
                self.ax2.add_patch(self.rect_sub)
                self.ax2.figure.canvas.draw()

####################################################################################################
######################## on Key ####################################################################

    def on_key(self, event):
    ### bind functions to certain key press ######

        #######################################
            #E Key - Subplot 2 grid position
        if event.key == 'e':
            if self.function_blocker == None:
            #activate dragable plot rectangle by pressing a
                self.make_rectangle = True
                self.grid_plot = True
                self.there_is_a_box = True
                self.function_blocker = 1

                print('Grid Plot selection active. Plot-Rectancle can be moved left/right by pressing A/D and up/down by pressing W/X')

        ########################################
            # U-Key - Fuse fish in selection of subplot 2
        if event.key == 'u':
            if self.is_subplot is True:
                self.select_grid = True
            print('\nGrid selection active!\nSelect by dragging over data in subplot.')

        #######################################
            #T Key - Fuse 2 fish
        if event.key == 't':
            if self.selection == 'pause' or self.selection =='multiple_selected_fish_t_one':
                self.make_rectangle = True
                self.selection = 2
                print('Select Second fish')
            if self.selection == None:
                if self.function_blocker == None:
                    print('\nSelect two fish by clicking on them:')
                    self.make_rectangle = True
                    self.fish_fusion = True
                    self.selection = 1
                    self.function_blocker = 2
                    self.confirm = None
        #############################################
            # L key
        if event.key == 'l':
            if self.l_activater == True:

                self.all_fish = []
                self.all_ch = []
                self.all_std = []
                self.all_mean = []
                self.all_max = []


                for idx in range(len(self.selected_fish_sign)):
                    f = self.selected_fish_sign[idx]
                    for i in range(len(self.fish_list[f])):

                            x = self.fish_list[f][i][0]
                            self.test = self.af_ident[f][self.index[idx]]

                            self.selected_index = np.where(self.af_ident[f][self.index[idx]] == self.fish_list[f][i][1])[0]
                            if len(self.selected_index) > 0:

                                self.index_new = self.index[idx][self.selected_index]
                                self.sel_sign = self.af_sign[f][self.index_new]
                            else:
                                self.sel_sign = []


                            if len(self.sel_sign) > 1:
                                self.std_sign = round(stc.stdev(self.sel_sign), 3)
                                self.all_std.extend([self.std_sign])
                                self.mean_sign = round(stc.mean(self.sel_sign), 3)
                                self.all_mean.extend([self.mean_sign])
                                self.all_max.extend([round(max(self.sel_sign),3)])
                                self.all_ch.extend([x])
                                self.all_fish.extend([f])

                            elif len(self.sel_sign) == 1:
                                self.std_sign = 0
                                self.all_std.extend([self.std_sign])
                                self.mean_sign = self.sel_sign
                                self.all_mean.extend([self.mean_sign])
                                self.all_fish.extend([f])
                                self.all_ch.extend([x])
                                self.all_max.extend([round(max(self.sel_sign),3)])

                for l in range(len(self.all_mean)):
                    print('Fish: ', self.all_fish[l], '. Ch: ', self.all_ch[l], '. Sign: Mean:', self.all_mean[l], ' and Stdev: ',self.all_std[l],' max amplitude: ',self.all_max[l])

                self.l_activater = None
    #######################################
            # Z Key - divide fish at selected point
        if event.key == 'z':

            if self.selection == None:
                if self.function_blocker == None:
                    self.make_rectangle = True
                    self.selection = 'pressed_z'
                    self.function_blocker = 3
                    print('\nSelect fish to be divided\n')

            if self.selection == 'selected_z':
                self.selection = 'click_z'
                print('\nPress on plot to get the point where the fish should get divided\n')

            if self.selection =='multiple_selected_fish':
                self.selection = 'click_z'
                print('\nPress on plot to get the point where the fish should get divided\n')


        #############################
        # Selection of individuel fish in case of multiple fish by going up or down
        # in index of selected fish by pressing A and D

         ################################################################
        # D button Key for single fish selection in Z and T choosing upwards index

        if event.key == 'd':
            if self.selection =='multiple_selected_fish_t_one':

                if self.selection_counter == self.sel_fish_idx_range:
                    self.selection_counter = 0
                else:
                    self.selection_counter += 1
                self.Yfish1 = self.af_freq_fish[self.selected_fish[self.selection_counter]]
                self.Xfish1 = self.af_dtime_fish[self.selected_fish[self.selection_counter]]

                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)



                self.fish_selection, = self.ax1.plot(self.Xfish1, self.Yfish1, marker='o', markerfacecolor='none',
                                                     markeredgecolor='yellow', markeredgewidth=0.5)
                self.ax1.figure.canvas.draw()
                self.fish1 = self.selected_fish[self.selection_counter]

                print('Fish Nr:',self.fish1,'. Correct first fish selected?\nPress T and choose second fish. \nAdjust selection by pressing A and D Key.\nor Cancel by pressing C\n')

            if self.selection == 'multiple_selected_fish_t_two':

                if self.selection_counter == self.sel_fish_idx_range:
                    self.selection_counter = 0
                else:
                    self.selection_counter += 1
                self.Yfish2 = self.af_freq_fish[self.selected_fish[self.selection_counter]]
                self.Xfish2 = self.af_dtime_fish[self.selected_fish[self.selection_counter]]

                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)

                self.fish_selection, = self.ax1.plot(self.Xfish2, self.Yfish2, marker='o', markerfacecolor='none',
                                                     markeredgecolor='yellow', markeredgewidth=0.5)
                self.ax1.figure.canvas.draw()
                self.fish2 = self.selected_fish[self.selection_counter]

                print('Fish Nr:',self.fish2,'. Correct second fish selected? \nPress Y to confirm. \nAdjust selection by pressing A and D Key.\nor Cancel by pressing C\n')


            if self.selection =='multiple_selected_fish':

                if self.selection_counter == self.sel_fish_idx_range:
                    self.selection_counter = 0
                else:
                    self.selection_counter += 1
                self.Yfish = self.af_freq_fish[self.selected_fish[self.selection_counter]]
                self.Xfish = self.af_dtime_fish[self.selected_fish[self.selection_counter]]

                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)



                self.fish_selection, = self.ax1.plot(self.Xfish, self.Yfish, marker='o', markerfacecolor='none',
                                                     markeredgecolor='yellow', markeredgewidth=0.5)
                self.ax1.figure.canvas.draw()
                self.sel_fish = self.selected_fish[self.selection_counter]
                print('Correct fish selected? Press Z and choose split point by clicking. \n Adjust selection by pressing A and D Key.\n or Cancel by pressing C\n')
        #####################
        # A Key for single fish selection in Z and T choosing downwards index
        ### movement fish1 in T #####
        if event.key == 'a':
            if self.selection == 'multiple_selected_fish_t_one':
                if self.selection_counter == 0:
                    self.selection_counter = self.sel_fish_idx_range
                else:
                    self.selection_counter -= 1

                self.Yfish1 = self.af_freq_fish[self.selected_fish[self.selection_counter]]
                self.Xfish1 = self.af_dtime_fish[self.selected_fish[self.selection_counter]]

                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)

                self.fish_selection, = self.ax1.plot(self.Xfish1, self.Yfish1, marker='o',linestyle=':', markerfacecolor='none',
                                                     markeredgecolor='yellow', markeredgewidth=0.5)
                self.ax1.figure.canvas.draw()
                self.fish1 = self.selected_fish[self.selection_counter]


                print(
                    'Fish Nr:',self.sel_fish,'Correct fish selected? Press T and choose split point by clicking. \n Adjust selection by pressing A and D Key.\n or Cancel by pressing C\n')


            ### selection of fish 2 in T
            if self.selection == 'multiple_selected_fish_t_two':
                if self.selection_counter == 0:
                    self.selection_counter = self.sel_fish_idx_range
                else:
                    self.selection_counter -= 1

                self.Yfish2 = self.af_freq_fish[self.selected_fish[self.selection_counter]]
                self.Xfish2 = self.af_dtime_fish[self.selected_fish[self.selection_counter]]

                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)

                self.fish_selection, = self.ax1.plot(self.Xfish2, self.Yfish2,linestyle=':', marker='o', markerfacecolor='none',
                                                     markeredgecolor='yellow', markeredgewidth=0.5)
                self.ax1.figure.canvas.draw()
                self.fish2 = self.selected_fish[self.selection_counter]

                print(
                    'Fish Nr:',self.fish2,'Correct second fish selected? Press Y to comfirm. \n Adjust selection by pressing A and D Key.\n or Cancel by pressing C\n')

            ######## selection of fish in Z
            if self.selection == 'multiple_selected_fish':
                if self.selection_counter == 0:
                    self.selection_counter = self.sel_fish_idx_range
                else:
                    self.selection_counter -= 1

                self.Yfish = self.af_freq_fish[self.selected_fish[self.selection_counter]]
                self.Xfish = self.af_dtime_fish[self.selected_fish[self.selection_counter]]

                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)

                self.fish_selection, = self.ax1.plot(self.Xfish, self.Yfish, marker='o', markerfacecolor='none',
                                                     markeredgecolor='yellow', markeredgewidth=0.5)
                self.ax1.figure.canvas.draw()
                self.sel_fish = self.selected_fish[self.selection_counter]
                print(
                    'Correct fish selected? Press Z and choose split point by clicking. \n Adjust selection by pressing A and D Key.\n or Cancel by pressing C\n')

        ########################################
        ###### Y Button - Confirm button  ######

        if event.key == 'y':

            ######### finishing z button. fuse fish data in fishlist and delete abritary fish.#########

            ## save fish file
            if self.selection == 'confirm_t' or self.selection =='multiple_selected_fish_t_two':
                self.old_fish_list = copy.copy(self.fish_list)
                self.old_ident = copy.copy(self.ident)

                self.fish_list[self.fish1].extend(self.fish_list[self.fish2])
               

                if self.fish_list.__class__ == list:
                    self.fish_list.pop(self.fish2)
                else:
                    self.fish_list = self.fish_list.tolist()
                    self.fish_list.pop(self.fish2)
                    #self.fish_list = np.array(self.fish_list)

                print('Fish fused!\n')
                self.selection = None

                self.function_blocker = None
                self.update_fish_list()
                self.save_function()
                self.fish1 = []
                self.fish2 = []


            if self.selection == 'confirm_z':
                self.old_fish_list = copy.copy(self.fish_list)
                self.old_ident = copy.copy(self.ident)

                self.fish_list_append_before = []
                self.fish_list_append_after = []
                self.idx_counter = -1
                self.delete_pair = []





                for fl in self.fish_list[self.sel_fish]:


                    self.fl_ = self.times[self.timeidx[fl[0]]]
                    self.fl_dt_time = self.fl_

                    self.fish_ident_idx = self.ident[fl[0]] == fl[1]

                    self.fish_time_before_point = self.fl_dt_time < self.point_x
                    self.fish_time_after_point = self.fl_dt_time >= self.point_x

                    self.fish_ident_idx_after_dot = np.where(self.fish_ident_idx & self.fish_time_after_point)[0]
                    self.fish_ident_idx_before_dot = np.where(self.fish_ident_idx & self.fish_time_before_point)[0]


                    if len(self.fish_ident_idx_after_dot) > 0:
                        if len(self.fish_ident_idx_before_dot) > 0:

                            self.delete_pair.append(fl)

                            if len(self.fish_ident_idx_before_dot) > 0:
                                self.highest_id += 1
                                self.ident[fl[0]][self.fish_ident_idx_before_dot] = self.highest_id
                                self.fish_list_append_before.append([fl[0], self.highest_id])

                                self.highest_id += 1
                                self.ident[fl[0]][self.fish_ident_idx_after_dot] = self.highest_id
                                self.fish_list_append_after.append([fl[0], self.highest_id])


                        else:
                                self.highest_id += 1
                                self.ident[fl[0]][self.fish_ident_idx_after_dot] = self.highest_id
                                self.fish_list_append_after.append([fl[0], self.highest_id])

                for d in self.delete_pair:
                    self.fish_list[self.sel_fish].remove(d)


                if self.fish_list.__class__ == list:

                    self.fish_list[self.sel_fish].extend(self.fish_list_append_before)
                    self.fish_list.append(self.fish_list_append_after)
                else:
                    self.fish_list = self.fish_list.tolist()
                    self.fish_list[self.sel_fish].extend(self.fish_list_append_before)
                    self.fish_list.append(self.fish_list_append_after)


                print('Fish divided')
                self.selection = None
                self.function_blocker = None
                self.update_fish_list()
                self.save_function()
            #######################
            ### confirm of grid plot U button
            if self.function_blocker != 2:
                if self.confirm == True:

                    self.old_fish_list = copy.copy(self.fish_list)
                    self.old_ident = copy.copy(self.ident)

                    self.new_fish_append= []

                    self.new_pop = []
                    self.delete_id = False
                    #self.highest_id += 1
                    #self.static_highest_id = self.highest_id

                    for idx in range(len(self.new_sel_fish)):
                        f = self.new_sel_fish[idx]
                        self.new_remove_fl = []
                        self.old_fish_append = []
                        for i in tqdm(range(len(self.fish_list[f]))):
                            self.ident_fish = self.fish_list[f][i][1]
                            ch =self.fish_list[f][i][0]

                            self.ident_idx_new_fl = []

                            if ch in self.new_unique_ch:

                                ### get index where and slected ident , X and y Values match
                                self.new_ident_index = np.where(self.ident[ch] == self.ident_fish)[0]
                                self.ident_idx_old_fl = self.new_ident_index.tolist()

                                for a in self.new_ident_index:
                                    for b in self.new_index[idx]:
                                        if self.af_freq_fish[f][b] == self.freq[ch][a]:

                                            if self.af_timeidx[f][b] == self.timeidx[ch][a]:
                                                self.ident_idx_new_fl.extend([a])
                                                if any(self.ident_idx_new_fl):
                                                    self.delete_id = True

                                if self.delete_id == True:



                                    self.ident_idx_new_fl = np.unique(self.ident_idx_new_fl)

                                    for rmo in self.ident_idx_new_fl:
                                        self.ident_idx_old_fl.remove(rmo)
                                    ### change id of selected pint to new highest
                                    self.highest_id += 1
                                    self.ident[ch][self.ident_idx_new_fl] = self.highest_id
                                    self.new_remove_fl.append([ch, self.ident_fish])
                                    self.new_fish_append.append([ch,self.highest_id])

                                    # change rest of fish list to another id and save new fl_pair to append to old fish
                                    if any(self.ident_idx_old_fl):
                                        self.highest_id += 1
                                        self.ident[ch][self.ident_idx_old_fl] = self.highest_id
                                        self.old_fish_append.append([ch,self.highest_id])

                                    self.delete_id = False

                        ## remove ald fl pairs
                        for pop in self.new_remove_fl:
                            self.fish_list[f].remove(pop)

                        # add still existing old fl pairs as new ones
                        self.fish_list[f].extend(self.old_fish_append)
                        # make list to delete fish if he is empty
                        if len(self.fish_list[f]) == 0:
                            self.new_pop.extend([f])
                    if self.fish_list.__class__ == list:

                        self.fish_list.append(self.new_fish_append)
                    else:
                        self.fish_list = self.fish_list.tolist()
                        self.fish_list.append(self.new_fish_append)

                        #self.fish_list = np.array(self.fish_list)

                    self.new_pop.reverse()

                    if len(self.new_pop) > 0:
                        for p in self.new_pop:
                            if self.fish_list.__class__ == list:
                                self.fish_list.pop(p)
                            else:
                                self.fish_list = self.fish_list.tolist()
                                self.fish_list.pop(p)
                                

                    print('\n Sucessfull fusion! \n')

                    self.update_fish_list()

                    self.save_function()



        ########################################
            #  C Button - cancel

        # cancels function button and resets canvas
        if event.key == 'c':
            if self.confirm == True:
                self.confrim = None
                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)
                self.ax1.figure.canvas.draw()
                self.selection = None
                self.make_rectangle = None
                self.function_blocker = None

            if self.undo == True:
                self.undo = None
                print('undo canceled')
                self.function_blocker = None
            if self.selection == 'pause' or self.selection == 'confirm_t' :
                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)
                self.ax1.figure.canvas.draw()
                self.selection = None
                self.make_rectangle = None
                self.function_blocker = None
            if self.selection == 'multiple_selected_fish' or self.selection == 'confirm_z' or self.selection == 'selected_z' or self.selection == 'click_z' :
                self.ax1.cla()
                for f in range(len(self.af_dtime_fish)):
                    self.freq_time_plot, = self.ax1.plot(self.af_dtime_fish[f][:], self.af_freq_fish[f][:], 'o',
                                                         picker=True, pickradius=5)
                self.ax1.figure.canvas.draw()
                self.selection = None
                self.make_rectangle = None
                self.function_blocker = None
        ########################################
            # O button -Rewert to old fish list and ident list
        if event.key == 'o':
            if self.undo == None:
                print('Pess O to confirm undo of last action otherwise press c for cancel\n')
                self.undo = True
            if self.undo == True:
                self.fish_list = self.old_fish_list
                self.ident = self.old_ident
                self.update_fish_list()
                self.save_function()
                self.function_blocker = None
                print('Change reverted \n')


        ########################################
            # Movement of rectancle by Keys

        if self.there_is_a_box is True :
            ###### move the rectangle with same size as before by pressing a certain button######
            # a/left, d/right, w/up and s/down


            if event.key == 'a':

                self.Xdiff = (self.Xmax-self.Xmin)*1/10
                self.Xmin -= self.Xdiff
                self.Xmax -= self.Xdiff

                self.adjust_plot = True



            if event.key == 'd':

                self.Xdiff = (self.Xmax - self.Xmin) * 1 / 10
                self.Xmin += self.Xdiff

                self.Xmax += self.Xdiff

                self.adjust_plot = True

            if event.key == 'w':

                self.Ydif = (self.Ymax - self.Ymin)/10
                self.Ymin += self.Ydif
                self.Ymax += self.Ydif
                self.adjust_plot = True
            if event.key == 'x':

                self.Ydif = (self.Ymax - self.Ymin) / 10
                self.Ymin -= self.Ydif
                self.Ymax -= self.Ydif
                self.adjust_plot = True

            if self.adjust_plot is True:
                self.rect.set_width(self.Xmax - self.Xmin)
                self.rect.set_height(self.Ymax - self.Ymin)
                self.rect.set_xy((self.Xmin, self.Ymin))
                self.ax1.figure.canvas.draw()
                self.adjust_plot = True

                ### adjust subplot 2
                self.fish_id_counterX = []
                self.fish_id_counterY = []
                self.Xindex = []
                self.Yindex = []
                self.index = []

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
                        self.X_index = np.array(self.indexX[0])
                        self.Xindex = self.Yindex[i][0][self.X_index]
                        self.index.append(self.Xindex)
                print('Identified fish', self.fish_id_counterX)
                # use index of Xindex in Yindex to find general index where everythign is true


                self.selected_fish = self.fish_id_counterX

                self.fish_color = []  # array with color of mainplot of a fish for subplot

                for c in self.selected_fish:
                    self.fish_color.append(self.ax1.get_lines()[c].get_color())

                ####### plotting selected data ############
                # plot selected grid i sublot 2
                self.ax2.cla()
                self.grid_pos_plot, = self.ax2.plot(x_grid, y_grid, marker='o', linestyle='None', color='black')
                self.ax2.set_ylabel('Distance [cm]')
                self.ax2.set_xlabel('Distance [cm]')
                # get channel and plot

                for i in range(len(self.index)):
                    self.ch_array = self.af_ch_list[self.selected_fish[i]][self.index[i]]
                    self.unique_ch = set(self.ch_array)
                    self.pos_indx = []

                    for x in self.unique_ch:

                        self.pos_indx.extend(np.where(x+1 == self.electrode_pos)[0])

                        self.random_number = self.last_random_number[i]

                        self.x_grid_sel = []
                        self.y_grid_sel = []

                        for j in self.pos_indx:
                            self.x_grid_sel.extend([x_grid[j]+self.random_number])
                            self.y_grid_sel.extend([y_grid[j]+self.random_number])



                    self.ax2.plot(self.x_grid_sel,self.y_grid_sel, marker='o', linestyle='None', markeredgewidth=2, color=self.fish_color[i])

                print('To select data points in Grid Plot to fuse them Press U\n')
                self.adjust_plot = None
                self.ax2.figure.canvas.draw()

##############################################################################
################################ main ########################################

if __name__ == "__main__":
    ########## loading data ############


    # load fishlist
    # auswahl datensatz a=0 21.10.19 a=1 22.10.19
    a = 0  # select record day
    # auswahl raw b = 0 bearbeitet b = 1
    b = 1    # select raw or processed data

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
        #### importiert electrode position grid ####
    from recording_grid_columbia_2019 import x_grid
    from recording_grid_columbia_2019 import y_grid
    from recording_grid_columbia_2019 import electrode_id_list
    ##### import fish data:#######
    if b == 0:
        filename = sorted(os.listdir('../../../kuehn/data/analysed_data'))[a]
        fish_list = np.load('fishlist' + filename + '.npy', allow_pickle=True)
        filepath = ('../../../kuehn/data/analysed_data/' + filename)
        fish_list = duplicate_hunter_fl(fish_list)
    elif b == 1:
        filename = sorted(os.listdir(save_path))[a]
        fish_list = np.load(save_path +'/'+ filename +'/fishlist.npy', allow_pickle=True)
        filepath = save_path +'/'+ filename
       


    dataextract = DataExtractor(fish_list, x_grid, y_grid, electrode_id_list, start_date, filepath)
    plt.show()
