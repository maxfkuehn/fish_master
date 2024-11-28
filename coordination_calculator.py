import datetime as dt
import math
import os
import random as rn
import statistics as stc
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


def get_highest_numbers(array, X):
    sorted_array = sorted(array, reverse=True)
    highest_numbers = sorted_array[:X]
    return highest_numbers


def line_equation(x, slope, intercept):
    if math.isinf(slope):

        print('Vertical Line')

        return 'vertical'
    else:
        return slope * x + intercept


def calculate_angle(slope):
    # Calculate the angle in radians using the inverse tangent function
    angle_rad = math.atan(slope)

    # Convert the angle from radians to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg


if __name__ == "__main__":
# Parameter Settings
    # auswahl datensatz a=0 21.10.19 a=1 22.10.19
    a = 0  # select record day
    b = 1  # data already calculated
    # min amplitude to count t [in mV]
    Vmin = 10  # [microV]
    # amount of highest amplitude counted
    X = 4
    # max allowed distance to highest amplitude electrode in cm,higher distance not counting
    distance = 300
    #not properly working electrodes
    forbidden_electrodes = [9,10,11, 12,13]

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



    ##### import fish data:#######

    # save path
    load_path = '/home/kuehn/Max_Masterarbeit/data/complete_processed_data'

    filename = sorted(os.listdir(load_path))[a]
    fish_list = np.load(load_path + '/' + filename + '/fishlist.npy', allow_pickle=True)
    filepath = load_path + '/' + filename
    save_date = record_day.replace('/', '')
    print(f'Processing: {save_date}')

    # arangement of elektrodes from bottomto top
    # get middle line of grid:
    middle_line_points = gmlc(x_grid, y_grid, electrode_id_list,
                              electrode_arrangement_list)  # list starts botton right

    # load data
    dic = flu(fish_list, filepath)

    ident_af = dic['ident_list']
    freq_af = dic['frequence_list']
    time_af = dic['time_list']
    time_idx_af = dic['time_idx_list']
    ch_af = dic['channel_list']
    sign_af = dic['sign_list']
    times = dic['record_time_array']
    sign_af_miV = dic['sign_list_microV']

    species_list_af = np.load(load_path + '/species_list_af.npy', allow_pickle=True)

    if a == 0:
        species_list = species_list_af[:len(ident_af)]
    else:
        species_list = species_list_af[len(species_list_af)-len(ident_af):]

    if b == 0:
        print('\nCalculating Fish Poistion:')
        # loop over every fish to get every timepoint of fish. For every timepoint find 4 highest electrodes.
        # Then calculate coordinates
        unique_time_points_af = []
        x_coordinate_af = []
        y_coordinate_af = []
        coordinates_af = []
        freq_values_af = []
        fish_nr = -1  # fish number
        species_list_corrected = []
        last_main_electrode_pos = []
        forbidden_electrode_counter = 0

        for time_list in time_af:  # loop over every fish and get theyre time list

            fish_nr += 1
            time_points = sorted(set(time_list))  # get list of unique time points
            x_value = []
            y_value = []
            frq_value = []
            coordinate_value = []
            time_value = []
            time_counter = -1
            for t_idx in time_points:  # look at everytime points for highest electrode
                idx_ch = np.where(time_af[fish_nr] == t_idx)[0]
                sign_array = np.asarray(sign_af_miV[fish_nr])
                # sign_array = sign_af_miV[np.array(time_list) == t_idx]
                highest_amplitudes = np.asarray(get_highest_numbers(sign_array[idx_ch], X))
                valid_indices = np.where(highest_amplitudes > Vmin)[0]
                time_counter += 1
                print('[',round(fish_nr/len(time_af)*100),'%] done: Loop per fish NR [',fish_nr,']: ', time_counter, 'of ', len(time_points), end='\r')

                if len(valid_indices) == 1:  # if only one channel is present or has minimal value
                    idx_highest = idx_ch[valid_indices][0]

                    electrode_idx = np.where(electrode_id_list == ch_af[fish_nr][idx_highest] + 1)[0][0]
                    x_coordinate = x_grid[electrode_idx]
                    y_coordinate = y_grid[electrode_idx]
                    x_value.extend([x_coordinate])
                    y_value.extend([y_coordinate])
                    coordinate_value.extend([x_coordinate, y_coordinate])
                    time_value.append(t_idx)
                    if isinstance(freq_af[fish_nr][idx_ch], np.ndarray):
                        frq_int=freq_af[fish_nr][idx_ch].astype(int)
                        frq_value.append(frq_int)
                    else:
                        frq_value.append(freq_af[fish_nr][idx_ch])
                    last_main_electrode_pos = [x_coordinate, y_coordinate]
                    last_time = t_idx

                elif len(valid_indices) > 1:  # if one or more channels meat condition of minimal V
                    # calculate weighetd x andy coordinates
                    amplitudes = highest_amplitudes[valid_indices]
                    idx_highest = idx_ch[valid_indices]
                    weigthed_x = []
                    weigthed_y = []
                    amplitudes_sum = []

                    counter = -1 #counter foor v_idx loop

                    loop_counter =-1 #loopcounter for distance checkto last highest electrode
                    # look for the highest amplitude of an electrode which is realisticly close to last highest to stop false data
                    mask_idx = 0#index to shrten list
                    if time_counter > 0:
                        time_diff = t_idx - last_time

                    if time_counter > 0 and time_diff < 10:
                        for indx in idx_highest:
                            el_idx = np.where(electrode_id_list == ch_af[fish_nr][indx] + 1)[0][0]

                            pos_el = [x_grid[el_idx], y_grid[el_idx]]

                            distance_last_he = ((last_main_electrode_pos[0] - pos_el[0]) ** 2 + (
                            last_main_electrode_pos[1] - pos_el[1]) ** 2) ** (1 / 2)

                            #Fprint(distance_last_he)

                            if distance_last_he < distance:
                                break

                            elif distance_last_he >= distance:
                                mask_idx =+ 1

                    if mask_idx < len(amplitudes):
                        amplitudes = amplitudes[mask_idx:]
                        idx_highest = idx_highest[mask_idx:]

                        if len(amplitudes) > 1:

                            highest_amplitude_electrode = ch_af[fish_nr][idx_highest[0]] + 1
                            high_el_idx = np.where(electrode_id_list == highest_amplitude_electrode)[0][0]
                            position_highest_eletrode = [x_grid[high_el_idx], y_grid[high_el_idx]]
                            time_value.extend([t_idx])

                            if isinstance(freq_af[fish_nr][idx_ch], np.ndarray):
                                frq_int = freq_af[fish_nr][idx_ch].astype(int)
                                frq_value.append(frq_int)
                            else:
                                frq_value.append(freq_af[fish_nr][idx_ch])
                            #fig = plt.figure()
                            #plt.plot(x_grid, y_grid, 'o', color='black')
                            #plt.plot(position_highest_eletrode[0], position_highest_eletrode[1], 'o', color='red')
                            #print('new')
                            for v_idx in idx_highest:  # get weighted coordinates,multiply coordinates by squareroot of amplitude

                                counter += 1 #loop/index counter
                                electrode_idx = np.where(electrode_id_list == (ch_af[fish_nr][v_idx] + 1))[0][0]
                                   # because channel numbers start at 1 and channel list index at 0
                                el_pos = [x_grid[electrode_idx],y_grid[electrode_idx]]

                                distance_to_main_electrode = ((position_highest_eletrode[0] - el_pos[0]) ** 2 + (position_highest_eletrode[1] - el_pos[1]) ** 2) ** (1 / 2) #main electrode is the electrode with biggest amplitude

                                if distance > distance_to_main_electrode:

                                    calculate_x = amplitudes[counter] ** (1 / 2) * x_grid[electrode_idx]
                                    weigthed_x.extend([calculate_x])
                                    calculate_y = amplitudes[counter] ** (1 / 2) * y_grid[electrode_idx]
                                    weigthed_y.extend([calculate_y])

                                    amplitudes_sum.append(amplitudes[counter])

                                    #if el_pos != position_highest_eletrode:
                                        #plt.plot(el_pos[0], el_pos[1], 'o', color='y')
                                    #print(distance_to_main_electrode)
                                    #print('ch is:', ch_af[fish_nr][v_idx] + 1)

                            #print('freqis:',freq_af[fish_nr][idx_highest])

                            #print(counter)

                            #plt.show()

                            #plt.close(fig)

                            final_x = sum(weigthed_x) / sum(np.asarray(amplitudes_sum) ** (1 / 2))
                            x_value.append(final_x)

                            final_y = sum(weigthed_y) / sum(np.asarray(amplitudes_sum) ** (1 / 2))
                            y_value.append(final_y)

                            last_main_electrode_pos = position_highest_eletrode
                            last_time = t_idx
                        else:

                            electrode_idx = np.where(electrode_id_list == ch_af[fish_nr][idx_highest] + 1)[0][0]
                            x_coordinate = x_grid[electrode_idx]
                            y_coordinate = y_grid[electrode_idx]
                            x_value.extend([x_coordinate])
                            y_value.extend([y_coordinate])
                            coordinate_value.extend([x_coordinate, y_coordinate])
                            time_value.extend([t_idx])
                            if isinstance(freq_af[fish_nr][idx_ch], np.ndarray):
                                frq_int = freq_af[fish_nr][idx_ch].astype(int)
                                frq_value.append(frq_int)
                            else:
                                frq_value.append(freq_af[fish_nr][idx_ch])
                            # time_value.append(t_idx)

                            last_main_electrode_pos = [x_coordinate, y_coordinate]
                            last_time = t_idx

            if len(x_value) > 60:  #if less then 60 data points ignore fish/throw out
                forbidden_electrodes_index = np.where(np.isin(electrode_id_list,forbidden_electrodes))[0]

                start_idx = 0
                end_idx = -1
                #check if first and/or last electrode os oneof the forbidden ones. Del first three and/or last three if yes.


                if x_value[0] == 0 and y_grid[forbidden_electrodes_index[-1]] <= y_value[0] and  y_value[0] <= y_grid[forbidden_electrodes_index[0]]:
                    start_idx = 4


                if x_value[-1] == 0 and y_grid[forbidden_electrodes_index[-1]] <= y_value[-1] and  y_value[-1] <= y_grid[forbidden_electrodes_index[0]]:
                    end_idx = - 5



                x_coordinate_af.append(x_value[start_idx:end_idx])
                y_coordinate_af.append(y_value[start_idx:end_idx])
                unique_time_points_af.append(time_value[start_idx:end_idx])
                freq_values_af.append(frq_value[start_idx:end_idx])
                species_list_corrected.append(species_list[fish_nr])

        # delete single data pints with no other point +- 10s away:



        np.save(load_path + record_day + '/x_coordinates.npy', np.array(x_coordinate_af, dtype=object))
        np.save(load_path + record_day + '/y_coordinates.npy', np.array(y_coordinate_af, dtype=object))
        np.save(load_path + record_day + '/times_coordinate.npy', np.array(unique_time_points_af, dtype=object))
        np.save(load_path + record_day + '/freq_values_movement.npy', np.array(freq_values_af, dtype=object))
        np.save(load_path + record_day + '/species_list.npy', np.array(species_list_corrected, dtype=object))
        print('Done!')

        x_coordinates = x_coordinate_af
        y_coordination = y_coordinate_af
        timepoints = unique_time_points_af
        freq_values = freq_values_af
        species_list = species_list_corrected
        # Project the coordinates of themiddlelinesof grid
    else:
        x_coordinates = np.load(load_path + '/' + filename + '/x_coordinates.npy', allow_pickle=True)
        y_coordination = np.load(load_path + '/' + filename + '/y_coordinates.npy', allow_pickle=True)
        timepoints = np.load(load_path + '/' + filename + '/times_coordinate.npy', allow_pickle=True)
        freq_values = np.load(load_path + '/' + filename + '/freq_values_movement.npy', allow_pickle=True)
        species_list = np.load(load_path + '/' + filename + '/species_list.npy' , allow_pickle=True)
        # calculate the line of each straight partof the grid to project onto
        # get middle grid
    middle_line_points = gmlc(x_grid, y_grid, electrode_id_list, electrode_arrangement_list)
    x_coords = [point[0] for point in middle_line_points]
    y_coords = [point[1] for point in middle_line_points]

    # parts go from coordinates in middlelist with idx [0:4],[4:5], [5:8] and [8:15]
    # calculate line by calculating m =(y2-y1)/(x2-x1)
    m4 = ((y_coords[4] - y_coords[0]) / (
            x_coords[4] - x_coords[0]))
    m3 = ((y_coords[5] - y_coords[4]) / (
            x_coords[5] - x_coords[4]))
    m2 = ((y_coords[8] - y_coords[5]) / (
            x_coords[8] - x_coords[5]))
    m1 = ((y_coords[15] - y_coords[8]) / (
            x_coords[15] - x_coords[8]))

    # caluclate b
    intercept4 = middle_line_points[4][1] - m4 * middle_line_points[4][0]
    intercept3 = middle_line_points[5][1] - m3 * middle_line_points[5][0]
    intercept2 = middle_line_points[8][1] - m2 * middle_line_points[8][0]
    intercept1 = middle_line_points[4][1] - m1 * middle_line_points[4][0]

    # calculate sectorlines, which seperate line sections. Coordinates of fish are projected onlyon line of each sector, sector points ee above
    # sector line is the bisector of the angle of the intersecting lines
    # calculate all angles of middle lines
    angle_l1 = calculate_angle(m1)  # ° angle of line 1 towards x achsis calculated by slope
    angle_l2 = calculate_angle(m2)  # ° angle of line 2 towards x achsis calculated by slope
    angle_l3 = calculate_angle(m3)  # ° angle of line 3 towards x achsis calculated by slope
    angle_l4 = calculate_angle(m4)  # ° angle of line 4 towards x achsis calculated by slope
    # bewteen l1:l2 :
    # calculate section shadow bountry:
    shadow1_angle1 = 90 - abs(angle_l1)  # angle of shadow line limit 1
    shadow1_angle2 = 90 - abs(angle_l2)  # angle of shadow line limit 2

    shadow1_m1 = math.tan(math.radians(shadow1_angle1))  # shadowslope
    shadow1_m2 = math.tan(math.radians(shadow1_angle2))

    shadow1_b1 = y_coords[8] - shadow1_m1 * x_coords[8]
    shadow1_b2 = y_coords[8] - shadow1_m2 * x_coords[8]
    # calclulate bisector
    angle_bis1_l2 = (180 - abs((angle_l2 - angle_l1))) / 2  # angle of line 4 towards bisector line
    angle_bis1 = angle_bis1_l2  # angle bisector 1 to x achsis. Because sector 1 is 90° to x-axis its also counting for x axis

    sector_m1 = math.tan(math.radians(angle_bis1))
    sector_b1 = y_coords[8] - sector_m1 * x_coords[8]

    # bewteen l2:l3' :

    # calculate section shadow bountry:
    shadow2_angle1 = 90 - abs(angle_l2)  # angle of shadow line limit 1
    shadow2_angle2 = 90 - abs(angle_l3)  # angle of shadow line limit 2

    shadow2_m1 = math.tan(math.radians(shadow2_angle1))
    shadow2_m2 = math.tan(math.radians(shadow2_angle2))

    shadow2_b1 = y_coords[5] - shadow2_m1 * x_coords[5]
    shadow2_b2 = y_coords[5] - shadow2_m2 * x_coords[5]

    # calclulate bisector
    # θ = θ2 - θ1
    # angle of line 4 towards bisector line
    angle_bis2_l3 = (180 - (abs(angle_l3 - angle_l2))) / 2  # angle of line 4 towards bisector line
    angle_bis2 = angle_bis2_l3 + angle_l3

    sector_m2 = math.tan(math.radians(angle_bis2))
    sector_b2 = y_coords[5] - sector_m2 * x_coords[5]

    # bewteen l3:l4 :
    # calculate section shadow bountry:
    shadow3_angle1 = 90 - abs(angle_l3)  # angle of shadow line limit 1
    shadow3_angle2 = 90 - abs(angle_l4)  # angle of shadow line limit 2

    shadow3_m1 = math.tan(math.radians(shadow3_angle1))
    shadow3_m2 = math.tan(math.radians(shadow3_angle2))

    shadow3_b1 = y_coords[4] - shadow3_m1 * x_coords[4]
    shadow3_b2 = y_coords[4] - shadow3_m2 * x_coords[4]

    # calclulate bisector
    angle_bis3_l4 = (180 - (angle_l4 - angle_l3)) / 2  # angle of line 4 towards bisector line
    angle_bis3 = angle_bis3_l4 + angle_l4

    sector_m3 = math.tan(math.radians(angle_bis3))
    sector_b3 = y_coords[4] - sector_m3 * x_coords[4]

    # calculate projection:

    # Formula: Projection = Point on line + ((Point - Point on line) x Unit vector) * Unit vector

    # calculate unit vector of each line
    # Formula Unit vector = [1/||[1, m]||, m/||[1, m]||]
    # Vec 1 not neededline is parralel to y axis
    # u_vec1 = [(1/(1**2+m1**2)**(1/2)),(m1/(1**2+m1**2)**(1/2))]
    u_vec2 = np.array([(1 / (1 ** 2 + m2 ** 2) ** (1 / 2)), (m2 / (1 ** 2 + m2 ** 2) ** (1 / 2))])
    u_vec3 = np.array([(1 / (1 ** 2 + m3 ** 2) ** (1 / 2)), (m3 / (1 ** 2 + m3 ** 2) ** (1 / 2))])
    u_vec4 = np.array([(1 / (1 ** 2 + m4 ** 2) ** (1 / 2)), (m4 / (1 ** 2 + m4 ** 2) ** (1 / 2))])

    # get point on line
    # point1 = [0,line_equation(0,m1,intercept1)]
    point2 = [100, line_equation(100, m2, intercept2)]
    point3 = [200, line_equation(200, m3, intercept3)]
    point4 = [300, line_equation(300, m4, intercept4)]

    projection_x = []
    projection_y = []
    af_section = []

    #calculate projection of electrodes onto middle line
    projection_el = [] # pairs [x,y]
    sector_el = []

    for el in electrode_arrangement_list:
        projection = []
        el_sector = []
        idx_el = np.where(np.isin(electrode_id_list, el))[0][0]

        x_p = x_grid[idx_el]
        y_p = y_grid[idx_el]

        #sector 1
        if el == 30:
            projection.append(x_coords[5])
            projection.append(y_coords[5])
            el_sector.append('p')
        elif y_p > line_equation(x_p, sector_m1, sector_b1):

            projection.extend([middle_line_points[-1][0]])
            projection.extend([y_p])
            el_sector.append('s1')
        #sector 2
        elif y_p < line_equation(x_p, sector_m1, sector_b1) and y_p > line_equation(x_p, sector_m2, sector_b2):

            point_vec_el = np.array([x_p, y_p]) - np.array(point2)

            calc_x = point2[0] + (np.dot(point_vec_el, u_vec2)) * u_vec2[0]
            calc_y = point2[1] + (np.dot(point_vec_el, u_vec2)) * u_vec2[1]

            projection.append(calc_x)
            projection.append(calc_y)
            el_sector.append('s2')
        #sector 3
        elif y_p < line_equation(x_p, sector_m2, sector_b2) and y_p > line_equation(x_p, sector_m3, sector_b3):

            point_vec_el = np.array([x_p, y_p]) - np.array(point3)

            calc_x = point3[0] + (np.dot(point_vec_el, u_vec3)) * u_vec3[0]
            calc_y = point3[1] + (np.dot(point_vec_el, u_vec3)) * u_vec3[1]

            projection.append(calc_x)
            projection.append(calc_y)
            el_sector.append('s3')

        # sector 4
        elif y_p < line_equation(x_p, sector_m3, sector_b3):

            point_vec_el = np.array([x_p, y_p]) - np.array(point4)

            calc_x = point4[0] + (np.dot(point_vec_el, u_vec4)) * u_vec4[0]
            calc_y = point4[1] + (np.dot(point_vec_el, u_vec4)) * u_vec4[1]

            projection.append(calc_x)
            projection.append(calc_y)
            el_sector.append('s4')

        sector_el.append(el_sector)
        projection_el.append(projection)


    #calculate projetion of fish position
    print('\n Calculation Projection:')

    for idx in tqdm(range(len(x_coordinates))):
        zip_coordinates = zip(x_coordinates[idx], y_coordination[idx])
        pro_x = []
        pro_y = []
        section = []

        for x_co, y_co in zip_coordinates:

            # on sectionborder 1

            if y_co == line_equation(x_co, sector_m1, sector_b1):
                pro_x.extend([x_coords[8]])
                pro_y.extend([y_coords[8]])
                section.extend(['p3'])
            # on sectionborder 2
            elif y_co == line_equation(x_co, sector_m2, sector_b2):
                pro_x.extend([x_coords[5]])
                pro_y.extend([y_coords[5]])
                section.extend(['p2'])
            # on sectionlborder 3
            elif y_co == line_equation(x_co, sector_m3, sector_b3):
                pro_x.extend([x_coords[4]])
                pro_y.extend([y_coords[4]])
                section.extend(['p1'])
            # on section 1

            elif y_co > line_equation(x_co, sector_m1, sector_b1):
                # check if point is in sector shadow
                if x_co < x_coords[8] and line_equation(x_co, shadow1_m1, shadow1_b1) > y_co and y_co > line_equation(
                        x_co, shadow1_m2, shadow1_b2):
                    pro_x.extend([x_coords[8]])
                    pro_y.extend([y_coords[8]])
                    section.extend(['p3'])
                else:
                    # line in this section is parallel to y axis -> projection y=y
                    pro_x.extend([middle_line_points[-1][0]])
                    pro_y.extend([y_co])
                    section.extend(['s4'])

            # on section 2
            elif y_co < line_equation(x_co, sector_m1, sector_b1) and y_co > line_equation(x_co, sector_m2, sector_b2):

                # check if point i shafow and  projectet on middleline
                if x_co < x_coords[8] and line_equation(x_co, shadow1_m1, shadow1_b1) > y_co and y_co > line_equation(
                        x_co, shadow1_m2, shadow1_b2):
                    pro_x.extend([x_coords[8]])
                    pro_y.extend([y_coords[8]])
                    section.extend(['p3'])
                # check if it is in shadow
                elif x_co < x_coords[5] and line_equation(x_co, shadow2_m1, shadow2_b1) > y_co and y_co > line_equation(
                        x_co, shadow2_m2, shadow2_b2):
                    pro_x.extend([x_coords[5]])
                    pro_y.extend([y_coords[5]])
                    section.extend(['p2'])
                else:
                    # Formula: Projection = Point on line + (Point - Point on line) * Unit vector * Unit vector
                    point_vec = np.array([x_co, y_co]) - np.array(point2)
                    calc_x = [point2[0] + (np.dot(point_vec, u_vec2)) * u_vec2[0]]
                    calc_y = [point2[1] + (np.dot(point_vec, u_vec2)) * u_vec2[1]]
                    pro_x.extend(calc_x)
                    pro_y.extend(calc_y)
                    section.extend(['s3'])
            # on section 3
            elif y_co < line_equation(x_co, sector_m2, sector_b2) and y_co > line_equation(x_co, sector_m3, sector_b3):
                # Formula: Projection = Point on line + (Point - Point on line) * Unit vector * Unit vector
                if x_co < x_coords[5] and line_equation(x_co, shadow2_m1, shadow2_b1) > y_co and y_co > line_equation(
                        x_co, shadow2_m2, shadow2_b2):
                    pro_x.extend([x_coords[5]])
                    pro_y.extend([y_coords[5]])
                    section.extend(['p2'])
                elif x_co < x_coords[4] and line_equation(x_co, shadow3_m1, shadow3_b1) > y_co and y_co > line_equation(
                        x_co, shadow3_m2, shadow3_b2):
                    pro_x.extend([x_coords[4]])
                    pro_y.extend([y_coords[4]])
                    section.extend(['p1'])
                else:
                    point_vec = np.array([x_co, y_co]) - np.array(point3)
                    calc_x = [point3[0] + (np.dot(point_vec, u_vec3)) * u_vec3[0]]
                    calc_y = [point3[1] + (np.dot(point_vec, u_vec3)) * u_vec3[1]]
                    pro_x.extend(calc_x)
                    pro_y.extend(calc_y)
                    section.extend(['s2'])
            # on section 4
            elif y_co < line_equation(x_co, sector_m3, sector_b3):
                # Formula: Projection = Point on line + (Point - Point on line) * Unit vector * Unit vector
                if x_co < x_coords[4] and line_equation(x_co, shadow3_m1, shadow3_b1) > y_co and y_co > line_equation(
                        x_co, shadow3_m2, shadow3_b2):
                    pro_x.extend([x_coords[4]])
                    pro_y.extend([y_coords[4]])
                    section.extend(['p1'])
                else:
                    point_vec = np.array([x_co, y_co]) - np.array(point4)
                    calc_x = [point4[0] + (np.dot(point_vec, u_vec4)) * u_vec4[0]]
                    calc_y = [point4[1] + (np.dot(point_vec, u_vec4)) * u_vec4[1]]
                    pro_x.extend(calc_x)
                    pro_y.extend(calc_y)
                    section.extend(['s1'])

        projection_x.append(pro_x)
        projection_y.append(pro_y)
        af_section.append(section)
    # to norm projections todinstance , calculate distance betwen pointsand start point
    # get dist to middle and end points

    start_point = projection_el[1] #start point first electrode in the bottom
    sec_point_1 = middle_line_points[4]
    sec_point_2 = middle_line_points[5]
    sec_point_3 = middle_line_points[8]
    # calculate distance with pytagoras  d=√((x_2-x_1)²+(y_2-y_1)²)

    dist_sec_point_1 = ((start_point[0] - sec_point_1[0]) ** 2 + (start_point[1] - sec_point_1[1]) ** 2) ** (1 / 2)
    dist_sec_point_2 = ((sec_point_1[0] - sec_point_2[0]) ** 2 + (sec_point_1[1] - sec_point_2[1]) ** 2) ** (1 / 2)
    dist_sec_point_3 = ((sec_point_2[0] - sec_point_3[0]) ** 2 + (sec_point_2[1] - sec_point_3[1]) ** 2) ** (1 / 2)

    #projection of channel position onto middle line



    # smoothing of the calculated x and y data


    fish_list_movement_distance = []

    for fish_number in range(len(projection_x)):
        fish_movement = []
        for idx_f in range(len(projection_x[fish_number])):
            # when in section 1
            if af_section[fish_number][idx_f] == 's1':
                point_dist_start = ((start_point[0] - projection_x[fish_number][idx_f]) ** 2 +
                                    (start_point[1] - projection_y[fish_number][idx_f]) ** 2) ** (1 / 2)

                fish_movement.extend([point_dist_start])
            # when in section 2
            elif af_section[fish_number][idx_f] == 's2':
                point_dist_start = ((sec_point_1[0] - projection_x[fish_number][idx_f]) ** 2 +
                                    (sec_point_1[1] - projection_y[fish_number][idx_f]) ** 2) ** (
                                           1 / 2) + dist_sec_point_1

                fish_movement.extend([point_dist_start])

            elif af_section[fish_number][idx_f] == 's3':
                point_dist_start = ((sec_point_2[0] - projection_x[fish_number][idx_f]) ** 2 +
                                    (sec_point_2[1] - projection_y[fish_number][idx_f]) ** 2) ** (
                                           1 / 2) + dist_sec_point_1 + dist_sec_point_2

                fish_movement.extend([point_dist_start])




            elif af_section[fish_number][idx_f] == 's4':
                point_dist_start = ((sec_point_3[0] - projection_x[fish_number][idx_f]) ** 2 +
                                    (sec_point_3[1] - projection_y[fish_number][idx_f]) ** 2) ** (
                                           1 / 2) + dist_sec_point_1 + dist_sec_point_2 + dist_sec_point_3
                fish_movement.extend([point_dist_start])

            elif af_section[fish_number][idx_f] == 'p1':
                point_dist_start = dist_sec_point_1
                fish_movement.extend([point_dist_start])

            elif af_section[fish_number][idx_f] == 'p2':

                point_dist_start = dist_sec_point_1 + dist_sec_point_2
                fish_movement.extend([point_dist_start])

            elif af_section[fish_number][idx_f] == 'p3':
                point_dist_start = dist_sec_point_1 + dist_sec_point_2 + dist_sec_point_3
                fish_movement.extend([point_dist_start])

        fish_list_movement_distance.append(fish_movement)

    #electrode position as distance
    dist_el = []
    for pro in range(len(projection_el)):
        if sector_el[pro] == ['p']:
            dist = dist_sec_point_1+dist_sec_point_2+dist_sec_point_3
            dist_el.append(dist)

        elif sector_el[pro] == ['s4']:

            dist = ((start_point[0] - projection_el[pro][0]) ** 2 +
                                (start_point[1] - projection_el[pro][1]) ** 2) ** (1 / 2)
            dist_el.append(dist)

        elif sector_el[pro] == ['s3']:
            dist = ((sec_point_1[0] - projection_el[pro][0]) ** 2 +
                                (sec_point_1[1] - projection_el[pro][1]) ** 2) ** (1 / 2) +dist_sec_point_1
            dist_el.append(dist)
        elif sector_el[pro] == ['s2']:
            dist = ((sec_point_2[0] - projection_el[pro][0]) ** 2 +
                                (sec_point_2[1] - projection_el[pro][1]) ** 2) ** (1 / 2) +dist_sec_point_1 +dist_sec_point_2
            dist_el.append(dist)
        elif sector_el[pro] == ['s1']:
            dist = ((sec_point_3[0] - projection_el[pro][0]) ** 2 +
                                (sec_point_3[1] - projection_el[pro][1]) ** 2) ** (1 / 2) +dist_sec_point_1 + dist_sec_point_2 +dist_sec_point_3
            dist_el.append(dist)
    embed()
    # kick bad electrodes once and for all
    cleaned_movement = []
    clean_x_coordinates = []
    clean_y_coordinates = []
    clean_timepoints = []
    clean_freq = []
    clean_species = []
    #smooth the movement
    # Smoothing parameters
    window_size = 40
    polyorder = 4

    # Apply smoothing to each list in the list of lists
    smoothed_data = []
    f_counter = 0
    counter = -1
    for sublist in fish_list_movement_distance:
        counter += 1
        if len(sublist) >= 60:
            smoothed_movement = savgol_filter(sublist, window_length = window_size, polyorder = polyorder)
            smoothed_data.append(smoothed_movement)
            clean_species.append(species_list[counter])
    for list in timepoints:
        if len(list) >= 60:
            clean_timepoints.append(list)

    for list in freq_values:
        if len(list) >= 60:
            clean_freq.append(list)

    np.save(load_path + record_day + '/timepoints_smooth_movement', np.array(clean_timepoints, dtype=object))
    np.save(load_path + record_day + '/smoothed_fish_movement', np.array(smoothed_data, dtype=object))
    np.save(load_path + record_day + '/freq_values_movement', np.array(clean_freq, dtype=object))
    np.save(load_path + record_day + '/species_list_corrected', np.array(clean_species, dtype=object))
    print('Done!')
