import datetime as dt
import math
import os
import random as rn
import statistics as stc
import seaborn as sns
import pdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from IPython import embed
from tqdm import tqdm
from scipy.signal import savgol_filter
# load own functions
from fish_list_unpacker import fish_list_unpacker as flu
#### importiert electrode position grid ####
from recording_grid_columbia_2019 import x_grid
from recording_grid_columbia_2019 import y_grid
from recording_grid_columbia_2019 import electrode_id_list
from recording_grid_columbia_2019 import electrode_arrangement_list
from grid_middle_line_calculator import grid_middle_line_calculator as gmlc
import random



# Scriptdetects Migration direction (fish swimming through grid)
if __name__ == "__main__":
# Parameter Settings
    # auswahl datensatz a=0 21.10.19 a=1 22.10.19 or alls [0,1]
    selected = [0, 1]
    # distance limit to count as migration start:
    migration_up = 600
    migration_down = 1150


    # load fishlist
    #### date time ######
    # definiere Zeitpunkte der Datei als Datum (datetime):
    start_date_0 = dt.datetime(2019, 10, 21, 13, 25, 00)
    start_date_1 = dt.datetime(2019, 10, 22, 8, 13, 00)
    record_day_0 = '/2019-10-21-13_25'
    record_day_1 = '/2019-10-22-8_13'

    # data of both days list of lists. Each list is a day list[0] = startdate0 list[1] = startdate1
    both_days_movement = []
    both_days_freq  = []
    both_days_timepoints = []
    both_days_migration_up_times = []
    both_days_migration_down_times = []
    both_days_migration_up_dt = []
    both_days_migration_down_dt = []
    both_days_average_migration_speed_up= []
    both_days_average_migration_speed_down = []
    both_days_freq_up = []
    both_days_freq_down = []
    both_days_stationary = []
    both_days_day = [['21.10.19'],['22.10.19']]
    both_days_migration_ratio_std_up = []
    both_days_migration_not_alone_ratio_up = []
    both_days_migration_ratio_std_down = []
    both_days_migration_not_alone_ratio_down = []
    both_days_ratio_full_list_up = []
    both_days_ratio_full_list_down = []
    both_days_pairs_species_up = []
    both_days_pairs_species_down = []
    both_days_species_up = []
    both_days_species_down = []
    both_days_pairs_list_down = []
    both_days_pairs_list_up = []
    both_days_stationary_fish = []
    for a in selected:

        if a == 0:
            start_date = start_date_0
            record_day = record_day_0

        elif a == 1:
            start_date = start_date_1
            record_day = record_day_1

        ##### import fish data:#######
        # load path
        load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'
        # set sunrise and sunset time as dt
        if start_date.hour > 6:
            sunrise_time = dt.datetime(year=start_date.year, month=start_date.month, day=start_date.day+1, hour=5, minute=34)
        else:
            sunrise_time = dt.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=5, minute=34)
        sunset_time = dt.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=17, minute=54)

        print(f'Processing: {record_day}')

        # save path
        load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'

        # load movement data and time points
        movement_list = np.load(load_path + record_day + '/smoothed_fish_movement.npy', allow_pickle=True)
        time_points = np.load(load_path + record_day + '/timepoints_smooth_movement.npy', allow_pickle=True)
        freq_values = np.load(load_path + record_day + '/freq_values_movement.npy', allow_pickle=True)

        both_days_movement.append(movement_list)
        both_days_timepoints.append(time_points)
        both_days_freq.append(freq_values)

        species_list = np.load(load_path + record_day +'/species_list.npy', allow_pickle=True)

        up_migration = 0
        down_migration = 0
        stationary = 0
        fish_upwards = [] #list of indx in fishlist of fish moving upwards
        fish_downwards = [] #list of indx in fishlist of fish moving downwards
        fish_stationary = [] #list of indx in fishlist of fish cooming to or staying in grid


        #for every fish check last and first movement and look if fish ended on the other side of the grid or stayed

        for fish_nr in range(len(movement_list)):
            start_end_dif = movement_list[fish_nr][-3] - movement_list[fish_nr][2]

            if start_end_dif > 800 :
                up_migration += 1
                fish_upwards.append(fish_nr)

            elif start_end_dif < -800 :
                down_migration += 1
                fish_downwards.append(fish_nr)

            else:
                stationary += 1
                fish_stationary.append(fish_nr)



        # check for start point of Migration
        #start point upwards, defined when t0; x>400 with t-1,x<400 and take the last point


        up_migration_start_times = []
        down_migration_sart_times = []
        up_migration_end_time = []
        down_migration_end_time = []
        up_migartion_freq = []
        down_migration_freq = []
        average_speed_up = []
        average_speed_down = []
        species_up = []
        species_down = []
        both_days_stationary_fish.append(fish_stationary)

        for fish_up in fish_upwards:

            bottom_limit_up = np.where(movement_list[fish_up] >= migration_up)[0]
            indx_start_up = bottom_limit_up[np.where(movement_list[fish_up][bottom_limit_up-1] < migration_up)][-1] #last timepoint where y0 >= limit migration up and y-1<limit migration up


            top_limit_up = np.where(movement_list[fish_up] >= migration_down)[0]

            try:
                indx_stop_up = top_limit_up[
                        np.where(movement_list[fish_up][top_limit_up - 1] < migration_down)][-1] # last timepoint where y0 >= 400 and y-1<400
            except IndexError:
                embed()

            up_migration_start_times.append(time_points[fish_up][indx_start_up])
            fish_freq_up = np.concatenate(freq_values[fish_up])
            up_migartion_freq.append(fish_freq_up[indx_start_up])

            time_diff_up = abs(time_points[fish_up][indx_start_up] - time_points[fish_up][indx_stop_up])
            last_point_diff_up = abs(
                (movement_list[fish_up][indx_start_up] - movement_list[fish_up][indx_stop_up])) / 100
            speed_up = last_point_diff_up / time_diff_up*60
            average_speed_up.append(speed_up)
            up_migration_end_time.append(time_points[fish_up][indx_stop_up])
            species_up.append(species_list[fish_up])


        for fish_down in fish_downwards:

            limit_up = np.where(movement_list[fish_down] >= migration_down)[0]

            indx_start_down = limit_up[np.where(movement_list[fish_down][limit_up + 1] < migration_down)][
                    -1]  # last timepoint where y0 >= limit migration down and y-1< limit migration down


            end_limit_down = np.where(movement_list[fish_down] <= migration_up)[0]
            
            indx_stop_down = end_limit_down[
                np.where(movement_list[fish_down][end_limit_down - 1] > migration_up)][
                -1]  # last timepoint where y0 >= 400 and y-1<400

            down_migration_sart_times.append(time_points[fish_down][indx_start_down])
            down_migration_end_time.append(time_points[fish_down][indx_stop_down])

            time_diff = abs(time_points[fish_down][indx_start_down] - time_points[fish_down][indx_stop_down])

            last_point_diff = abs((movement_list[fish_down][indx_start_down] - movement_list[fish_down][indx_stop_down]))/100
            speed_down = last_point_diff/time_diff*60
            average_speed_down.append(speed_down)

            fish_freq_down = np.concatenate(freq_values[fish_down])
            down_migration_freq.append(fish_freq_down[indx_start_down])
            species_down.append(species_list[fish_down])

        up_dt_times = [start_date+dt.timedelta(seconds=t) for t in up_migration_start_times]
        down_dt_times = [start_date+dt.timedelta(seconds=t) for t in down_migration_sart_times]

          #checkking for swimming pairs
        pairs_migration_list_up = [] # list of lists , eachlist has id of fish and then id of pait partner secind [[id fish],[ids partner]
        pairs_migration_list_down = []  # list of lists , eachlist has id of fish and then id of pait partner secind [[id fish],[ids partner]
        pairs_distance_diff_up = [] # distance of fish to each migrated fish in same time frame
        pairs_distance_diff_down = []
        pairs_migration_timepoints_up = [] # time points of migrated fish
        pairs_migration_timepoints_down = []
        pairs_species_up = []
        pairs_species_down = []
        '''' -----------------------------------------------------------------------
                   UPWARDS, CHECKING FOR PAIRS
                   -------------------------------------------------------------------------'''
        # upwards
        for fish_up in range(len(fish_upwards)):

            fish_pairs = []
            start = up_migration_start_times[fish_up]
            end = up_migration_end_time[fish_up]
            zip_data = list(zip(time_points[fish_upwards[fish_up]], movement_list[fish_upwards[fish_up]]))
            sorted_zip = sorted(zip_data, key=lambda x: x[0])
            sorted_time = np.array([item[0] for item in sorted_zip])
            sorted_movement = np.array([item[1] for item in sorted_zip])

            m_idx = np.where((sorted_time >= start) & (sorted_time <= end))[0]

            migration_tp = sorted_time[m_idx]
            migration_distance = sorted_movement[m_idx]

            overlapping_fish = []

            for other_fish in range(len(fish_upwards)):
                start_of = up_migration_start_times[other_fish]
                end_of = up_migration_end_time[other_fish]

                if start_of < end and end_of > start and other_fish != fish_up:
                    overlapping_fish.append(fish_upwards[other_fish])



            if len(overlapping_fish) == 0:
                fish_overlap_distance = []
                animal_distance = []
                hit_idx = []
            else:
                for ov in overlapping_fish:
                    fish_overlap_distance = []

                    hit_idx = []
                    for mig_t, mig_d in zip(migration_tp,migration_distance):
                        hit_idx = np.where(time_points[ov] == mig_t)[0]
                        ov_movement = movement_list[ov]

                        if hit_idx.size > 0:
                            animal_distance = (ov_movement[hit_idx][0] - mig_d)
                            fish_overlap_distance.append(animal_distance)

                        else :
                            fish_overlap_distance.append(np.nan)

                    if len(fish_overlap_distance) > 0:
                        fish_pairs.append(fish_overlap_distance)
            if len(fish_pairs) > 0:
                pairs_migration_list_up.append([[fish_upwards[fish_up]], overlapping_fish])
                pairs_migration_timepoints_up.append(migration_tp)
                pairs_distance_diff_up.append(fish_pairs)

        pairs_check_per_fish_up = []
        pairs_check_added_list_up = []
        pairs_ratio_list_up = []
        pairs_all_fish_ratio_up = []
        pairs_species_up = []
        for fish_idx in range(len(pairs_distance_diff_up)):
            fish_pair = []
            ratio_info = []
            fish_number = pairs_migration_list_up[fish_idx][0]

            for pair in pairs_distance_diff_up[fish_idx]:
                # checking if distance is under or exactly 100 cm , electrode distance 1 m
                distance_check_list = [distance >= 100 for distance in pair]
                fish_together = np.asarray([int(value) for value in distance_check_list])
                fish_pair.append(fish_together)

            summed_pairs = np.sum(np.array(fish_pair),axis=0)
            pairs_check_added_list_up.append(summed_pairs)

            # Create a Dictoionary for ratio with percentage etc
            # Define the categories
            categories = ["0", "1", "2", "3+", ]

            # Create a list to hold the category labels
            category_labels = categories
            # create listto store categorys and infos
            fish_pair_ratio_info = []

            # Count the occurrences of each category
            for category in categories:
                if category == "3+":
                    count = np.count_nonzero(summed_pairs >= 3)
                else:
                    count = np.count_nonzero(summed_pairs == int(category))
                fish_pair_ratio_info.append({"fish": fish_number,"category": category, "count": count})
            # Calculate the total count
                total_count = len(summed_pairs)

            # Calculate and add ratios to the category information
            for category_dict in fish_pair_ratio_info:
                count = category_dict["count"]
                ratio = count / total_count
                category_dict["ratio"] = ratio
            pairs_ratio_list_up.append(fish_pair_ratio_info)
            pairs_species_up.append(species_list[fish_number])
        # Calculate the overall average of all fish pair numbers over all fish

        for cat_idx in range(len(categories)):

            ratio_fish = [inner_list[cat_idx]["ratio"] for inner_list in pairs_ratio_list_up]
            if cat_idx == 0:
                pairs_ratio_not_alone = 1 - np.array(ratio_fish)
            average_ratio = sum(ratio_fish)/len(pairs_ratio_list_up)
            std_pair = np.std(ratio_fish)
            pairs_all_fish_ratio_up.append({"day": record_day,"category": categories[cat_idx], "ratio": average_ratio, "std": std_pair})

        both_days_migration_ratio_std_up.append(pairs_all_fish_ratio_up)
        both_days_migration_not_alone_ratio_up.append(pairs_ratio_not_alone)
        both_days_ratio_full_list_up.append(pairs_ratio_list_up)
        both_days_pairs_species_up.append(pairs_species_up)
        '''' -----------------------------------------------------------------------
            DOWNWARDS: CHECKING FOR PAIRS
        -------------------------------------------------------------------------'''

        #same for down migration
        for fish_down in range(len(fish_downwards)):
            fish_pairs = []
            start = down_migration_sart_times[fish_down]
            end = down_migration_end_time[fish_down]
            zip_data = list(zip(time_points[fish_downwards[fish_down]], movement_list[fish_downwards[fish_down]]))
            sorted_zip = sorted(zip_data, key=lambda x: x[0])
            sorted_time = np.array([item[0] for item in sorted_zip])
            sorted_movement = np.array([item[1] for item in sorted_zip])

            m_idx = np.where((sorted_time >= start) & (sorted_time <= end))[0]

            migration_tp = sorted_time[m_idx]
            migration_distance = sorted_movement[m_idx]

            overlapping_fish = []

            for other_fish in range(len(fish_downwards)):
                start_of = down_migration_sart_times[other_fish]
                end_of = down_migration_end_time[other_fish]

                if start_of < end and end_of > start and other_fish != fish_down:
                    overlapping_fish.append(fish_downwards[other_fish])
            if len(overlapping_fish) == 0:
                fish_overlap_distance = []
                animal_distance = []
                hit_idx = []
            else:
                for ov in overlapping_fish:
                    fish_overlap_distance = []
                    for mig_t, mig_d in zip(migration_tp,migration_distance):
                        hit_idx = np.where(time_points[ov] == mig_t)[0]

                        ov_movement = movement_list[ov]

                        if hit_idx.size > 0:
                            animal_distance = (ov_movement[hit_idx][0] - mig_d)
                            fish_overlap_distance.append(animal_distance)

                        else :
                            fish_overlap_distance.append(np.nan)

                    if len(fish_overlap_distance) > 0:
                        fish_pairs.append(fish_overlap_distance)


            if len(overlapping_fish) > 0:
                pairs_migration_list_down.append([[fish_downwards[fish_down]], overlapping_fish])
                pairs_migration_timepoints_down.append(migration_tp)

                pairs_distance_diff_down.append(fish_pairs)

        pairs_check_per_fish_down = []
        pairs_check_added_list_down = []
        pairs_ratio_list_down = []
        pairs_all_fish_ratio_down = []
        pairs_species_down = []
        for fish_idx in range(len(pairs_distance_diff_down)):
            fish_pair = []
            ratio_info = []
            fish_number = pairs_migration_list_down[fish_idx][0]
            for pair in pairs_distance_diff_down[fish_idx]:
                # checking if distance is under or exactly 100 cm , electrode distance 1 m
                distance_check_list = [distance >= 100 for distance in pair]
                fish_together = np.asarray([int(value) for value in distance_check_list])
                fish_pair.append(fish_together)

            summed_pairs = np.sum(np.array(fish_pair), axis=0)
            pairs_check_added_list_up.append(summed_pairs)

            # Create a Dictoionary for ratio with percentage etc
            # Define the categories
            categories = ["0", "1", "2", "3+", ]

            # Create a list to hold the category labels
            category_labels = categories
            # create listto store categorys and infos
            fish_pair_ratio_info = []

            # Count the occurrences of each category
            for category in categories:
                if category == "3+":
                    count = np.count_nonzero(summed_pairs >= 3)
                else:
                    count = np.count_nonzero(summed_pairs == int(category))
                fish_pair_ratio_info.append({"fish": fish_number, "category": category, "count": count})
                # Calculate the total count
                total_count = len(summed_pairs)

            # Calculate and add ratios to the category information
            for category_dict in fish_pair_ratio_info:
                count = category_dict["count"]
                ratio = count / total_count
                category_dict["ratio"] = ratio
            pairs_ratio_list_down.append(fish_pair_ratio_info)
            pairs_species_down.append(species_list[fish_number])
        # Calculate the overall average of all fish pair numbers over all fish

        for cat_idx in range(len(categories)):

            ratio_fish = [inner_list[cat_idx]["ratio"] for inner_list in pairs_ratio_list_down]
            if cat_idx == 0:
                pairs_ratio_not_alone_down = 1 - np.array(ratio_fish)
            average_ratio = sum(ratio_fish) / len(pairs_ratio_list_down)
            std_pair = np.std(ratio_fish)
            pairs_all_fish_ratio_down.append(
                {"day": record_day, "category": categories[cat_idx], "ratio": average_ratio, "std": std_pair})

        both_days_migration_ratio_std_down.append(pairs_all_fish_ratio_down)
        both_days_migration_not_alone_ratio_down.append(pairs_ratio_not_alone_down)
        both_days_ratio_full_list_down.append(pairs_ratio_list_down)
        both_days_pairs_species_down.append(pairs_species_down)

        #list(itertools.combinations(ids, r=2))


        # Calculate the time differences in seconds
        sunrise_seconds = (sunrise_time - start_date).total_seconds()
        sunset_seconds = (sunset_time - start_date).total_seconds()

        both_days_migration_up_times.append(up_migration_start_times)
        both_days_migration_down_times.append(down_migration_sart_times)
        both_days_migration_up_dt.append(up_dt_times)
        both_days_migration_down_dt.append(down_dt_times)
        both_days_freq_up.append(up_migartion_freq)
        both_days_freq_down.append(down_migration_freq)
        both_days_average_migration_speed_up.append(average_speed_up)
        both_days_average_migration_speed_down.append(average_speed_down)
        both_days_species_up.append(species_up)
        both_days_species_down.append(species_down)
        both_days_pairs_list_down.append(pairs_migration_list_down)
        both_days_pairs_list_up.append(pairs_migration_list_up)
        #both_days_stationary.append()
    """""
   # Combine the datasets and corresponding datetimes
    all_boxplot_values = [up_migration_start_times, down_migration_sart_times]
    all_datetimes = [up_dt_times, down_dt_times]
    # get sunset and sunrise in seconds
    boxplot, ax1 = plt.subplots()
    # Create the box plots
    ax1.boxplot(all_boxplot_values, vert=False, labels=['up', 'down'])
    ax1.axvspan(sunset_seconds, sunrise_seconds, facecolor='black', alpha=0.2)
    # Flatten the nested list of datetimes to get all datetime values
    all_datetimes_flat = [dt for sublist in all_datetimes for dt in sublist]

    # Set the x-axis tick positions and labels using the corresponding datetimes
    tick_positions = all_datetimes_flat
    tick_labels = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in all_datetimes_flat]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    # Add labels for the axes and a title
    plt.xlabel('Datetime [%hours:minutes:seconds]')
    plt.title('Boxplots of upwards and downward Migration Start Times')"
    """""

    # migration start time and frquence of fish plot

    freq_time_plot,(ax,ay) = plt.subplots(2,1)
    #alpha_day = [1,0.5]

    fake_date = [[dt.datetime(1900, 1, 1, 13, 25, 00)],[dt.datetime(1900, 1, 1, 8, 13, 00)]]

    sunrise_time = fake_date[1][0]+dt.timedelta(seconds=sunrise_seconds)
    sunset_time = fake_date[1][0]+dt.timedelta(seconds=sunset_seconds)
    #colors up and down
    c_up = ['green','yellowgreen']
    c_down =['red','tomato']
    # Calculate the reference time for each day (midnight)
    reference_time = dt.datetime(1900, 1, 1, 0, 0, 0)

    for day in selected:

        label_day = both_days_day[day]
        diff_time = (fake_date[day][0] - reference_time).total_seconds()
        #make same day at date time
        up_times_nulled = [fake_date[day][0] + dt.timedelta(seconds=t) for t in both_days_migration_up_times[day]]
        down_times_nulled = [fake_date[day][0] + dt.timedelta(seconds=t) for t in both_days_migration_down_times[day]]
        ax.plot(down_times_nulled, both_days_freq_down[day],'o',color=c_down[day], label=f'down {label_day}')
        ax.plot(up_times_nulled, both_days_freq_up[day],'o',color=c_up[day], label=f'up {label_day}')
        up_boxplot = [diff_time + t for t in both_days_migration_up_times[day]]
        down_boxplot = [diff_time + t for t in both_days_migration_down_times[day]]
        ay.boxplot(up_boxplot, positions=[day], vert=False)
        ay.boxplot(down_boxplot, positions=[day], vert=False)


    ax.axvspan(sunrise_time, sunset_time, facecolor='black', alpha=0.2)
    # Format the x-axis datetime labels
    date_formatter = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.gcf().autofmt_xdate()
    # Retrieve the x-axis tick labels from the ax subplot
    ax_tick_labels = ax.get_xticklabels()
    # Apply the same tick labels to the ay subplot
    ay.set_xticklabels(ax_tick_labels)

    plt.legend(loc='upper right')
    plt.xlabel('Datetime [%hours:minutes:seconds]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Start time of migration and Frequency of each migrating fish')


    # average speed plot
    average_speed_plot, ax1 = plt.subplots()
    all_speed_up = np.concatenate(both_days_average_migration_speed_up)
    all_speed_down = np.concatenate(both_days_average_migration_speed_down)
    all_data = [all_speed_up, all_speed_down]

    ax1.boxplot(all_data, positions= [1,2], labels= ['up','down'], sym='' )

    plt.title('Average migration speed')
    plt.xlabel('Movement speed [m/min]')
    plt.show()
    embed()
    print('here')

    #saving pairs list
    np.save(load_path + '/both_days_pairs_list_up', np.array(both_days_pairs_list_up, dtype=object))
    np.save(load_path + '/both_days_pairs_list_down', np.array(both_days_pairs_list_down, dtype=object))
    #saving ratio
    np.save(load_path + '/both_days_ratio_up', np.array(both_days_migration_ratio_std_up, dtype=object))
    np.save(load_path + '/both_days_ratio_down', np.array(both_days_migration_ratio_std_down, dtype=object))
    np.save(load_path + '/both_days_ratio_not_alone_up', np.array(both_days_migration_not_alone_ratio_up, dtype=object))
    np.save(load_path + '/both_days_ratio_not_alone_down', np.array(both_days_migration_not_alone_ratio_down, dtype=object))
    np.save(load_path + '/both_days_pairs_species_down', np.array(both_days_pairs_species_down, dtype=object))
    np.save(load_path + '/both_days_pairs_species_up', np.array(both_days_pairs_species_up, dtype=object))
    #saving migration
    np.save(load_path + '/both_days_migration_times_up', np.array(both_days_migration_up_times, dtype=object))
    np.save(load_path + '/both_days_migration_times_down', np.array(both_days_migration_down_times, dtype=object))
    np.save(load_path + '/both_days_migration_dt_up', np.array(both_days_migration_up_dt, dtype=object))
    np.save(load_path + '/both_days_migration_dt_down', np.array(both_days_migration_down_dt, dtype=object))
    np.save(load_path + '/both_days_average_migration_speed_up', np.array(both_days_average_migration_speed_up, dtype=object))
    np.save(load_path + '/both_days_average_migration_speed_down', np.array(both_days_average_migration_speed_down, dtype=object))
    np.save(load_path + '/both_days_freq_up', np.array(both_days_freq_up, dtype=object))
    np.save(load_path + '/both_days_freq_down', np.array(both_days_freq_down, dtype=object))
    np.save(load_path + '/both_days_movement', np.array(both_days_movement, dtype=object))
    np.save(load_path + '/both_days_timepoints', np.array(both_days_timepoints, dtype=object))
    np.save(load_path + '/both_days_full_ratio_up', np.array(both_days_ratio_full_list_up, dtype=object))
    np.save(load_path + '/both_days_full_ratio_down', np.array(both_days_ratio_full_list_down, dtype=object))
    np.save(load_path + '/both_days_species_up', np.array(both_days_species_up, dtype=object))
    np.save(load_path + '/both_days_species_down', np.array(both_days_species_down, dtype=object))
    np.save(load_path + '/both_days_stationary_fish', np.array(both_days_stationary_fish, dtype=object))


