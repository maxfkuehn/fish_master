import math
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from grid_middle_line_calculator import grid_middle_line_calculator


def line_equation(x, slope, intercept):
    return slope * x + intercept


def calculate_angle(slope):
    # Calculate the angle in radians using the inverse tangent function
    angle_rad = math.atan(slope)

    # Convert the angle from radians to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg


# array of elektrode channels sorted by position in grid related to x,y coordinates
m0 = np.arange(8, 16)
m1 = np.arange(8)
s0 = np.arange(24, 32)
s1 = np.arange(16, 24)
ch_lists = m0[::-1], s0[::-1], m1[::-1], s1[::-1]
ch_pos_grid = np.concatenate(ch_lists, axis=0)

# xy coordinates of the grid
x_grid_2d = np.concatenate([np.zeros(16), np.ones(16)])
y_grid_2d = np.concatenate([np.arange(0, -16, -1), np.arange(0, -16, -1)])

# Abstand der Elektroden grid master rechten Seite
# Abstand zwischen Elektroden rechts ist aufegteilt in zwei Abstande, Mitte ist die position der Elektrode auf
# der Rechtenseite. Bsp. [55,55] -> elreode x-y haben Abtsand 110 und positionelektrode auf der anderen seite
# ist 55 von und 55 y

Grid_master_right_data = [70, 30], [70, 35], [65, 45], [55, 50], [50, 50], [60, 45], [55, 55]
degree_master_left = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
degree_master_right = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]

# calculate distance in master grid between elektrodes on the right row
end_diff_master_left = 40

master_distance_left = []

for i in range(len(Grid_master_right_data) - 1):
    a = [Grid_master_right_data[i][1], Grid_master_right_data[i + 1][0]]
    master_distance_left.append(a)
end_numbers = [Grid_master_right_data[-1][1], end_diff_master_left]
master_distance_left.append(end_numbers)
# Distancase slave grid, [0] dsitance and degree between slave and master

slave_distance_grid_data_left = [50, 50], [50, 50], [50, 50], [65, 35], [25, 75], [25, 75], [25, 75], [25, 75]
degree_slave_left = [30, 30], [30, 30], [30, 30], [30, 65], [65, 65], [65, 65], [65, 65], [65, 65]

slave_distance_grid_data_right = [70, 35], [50, 50], [50, 50], [65, 35], [75, 25], [75, 25], [75, 25], [75, 25]
degree_slave_right = [0, 30], [30, 30], [30, 30], [30, 65], [65, 65], [65, 65], [65, 65], [65, 65]

left_dist_list = [master_distance_left, slave_distance_grid_data_left]
left_dist_list = np.concatenate(left_dist_list)

left_degree_list = [degree_master_left, degree_slave_left]
left_degree_list = np.concatenate(left_degree_list)

right_dist_list = [Grid_master_right_data, slave_distance_grid_data_left]
right_dist_list = np.concatenate(right_dist_list)

right_degree_list = [degree_master_right, degree_slave_right]
right_degree_list = np.concatenate(right_degree_list)

y_coordinates_left = [0]
y_coordinates_right = [70]

x_coordinates_left = [0]
x_coordinates_right = [80]

#### calculating y coordinates based on yd ( degree of y point orientation from point befor) and yl ( point distance)
#### for right and left side of grid
for i in range(len(left_dist_list)):
    yd1 = np.cos(np.deg2rad(left_degree_list[i][0]))
    yl1 = np.int64(left_dist_list[i][0])
    a = yl1 * yd1
    yd2 = np.cos(np.deg2rad(left_degree_list[i][1]))
    yl2 = np.int64(left_dist_list[i][1])
    b = yl2 * yd2
    yl = y_coordinates_left[-1] - a - b
    if isinstance(yl, np.ndarray) is True:
        yl = yl[0]

    y_coordinates_left.extend([yl])

    yds1 = np.cos(np.deg2rad(left_degree_list[i][0]))
    yr1 = np.int64(left_dist_list[i][0])
    bb = yr1 * yd1
    yds2 = np.cos(np.deg2rad(left_degree_list[i][1]))
    yr2 = np.int64(left_dist_list[i][1])
    aa = yr2 * yds2
    yr = y_coordinates_right[-1] - aa - bb
    if isinstance(yr, np.ndarray) is True:
        yr = yr[0]
    y_coordinates_right.extend([yr])
## caluclationg x coordinates by distance and xd ( degree difference from pojt befor)
for i in range(len(left_dist_list)):

    xd1 = np.sin(np.deg2rad(left_degree_list[i][0]))
    x1 = np.int64(left_dist_list[i][0]) * xd1

    xd2 = np.sin(np.deg2rad(left_degree_list[i][1]))
    x2 = np.int64(left_dist_list[i][1]) * xd2

    xl = x_coordinates_left[-1] + x1 + x2
    if isinstance(xl, np.ndarray) is True:
        xl = xl[0]
    x_coordinates_left.extend([xl])

    xl1 = np.sin(np.deg2rad(right_degree_list[i][0]))
    xr1 = np.int64(right_dist_list[i][0]) * xd1

    xl2 = np.sin(np.deg2rad(right_degree_list[i][1]))
    xr2 = np.int64(right_dist_list[i][1]) * xd2

    xr = x_coordinates_right[-1] + xr1 + xr2
    if isinstance(xr, np.ndarray) is True:
        xr = xr[0]
    x_coordinates_right.extend([xr])

x_grid = []
x_grid.extend(x_coordinates_left)
x_grid.extend(x_coordinates_right)

y_grid = []
y_grid.extend(y_coordinates_left)
y_grid.extend(y_coordinates_right)

electrode_id_list = [16, 15, 14, 13, 12, 11, 10, 9, 32, 31, 30, 29, 28, 27, 26, 25, 8, 7, 6, 5, 4, 3, 2, 1, 24, 23, 22,
                     21, 20, 19, 18, 17]
electrode_arrangement_list = [25, 17, 26, 18, 27, 19, 28, 20, 29, 21, 30, 22, 31, 23, 32, 24, 9, 1, 10, 2, 11, 3, 12, 4,
                              13, 5, 14, 6, 15, 7, 16, 8]

np.save('elecrode_pos_list', electrode_id_list)
np.save('elecrode_arrangement', electrode_arrangement_list)
np.save('x_coordinates_grid', x_grid)
np.save('y_coordinates_grid', y_grid)

# get middle grid
middle_line_grid = grid_middle_line_calculator(x_grid, y_grid, electrode_id_list, electrode_arrangement_list)
x_coords = [point[0] for point in middle_line_grid]
y_coords = [point[1] for point in middle_line_grid]

electrode_id_list = np.array(electrode_id_list)

# calculate the line of each straight partof the grid to project onto
# get middle grid
middle_line_points = grid_middle_line_calculator(x_grid, y_grid, electrode_id_list, electrode_arrangement_list)
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
angle_bis1_l2 = (180-abs((angle_l2 - angle_l1))) / 2  # angle of line 4 towards bisector line
angle_bis1 = angle_bis1_l2   # angle bisector 1 to x achsis. Because sector 1 is 90° to x-axis its also counting for x axis


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
angle_bis2_l3 = (180-(abs(angle_l3 - angle_l2))) / 2  # angle of line 4 towards bisector line
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
angle_bis3_l4 = (180-(angle_l4 - angle_l3)) / 2  # angle of line 4 towards bisector line
angle_bis3 = angle_bis3_l4 + angle_l4

sector_m3 = math.tan(math.radians(angle_bis3))
sector_b3 = y_coords[4] - sector_m3 * x_coords[4]



# plot
fig, ax = plt.subplots()
x_m = np.array(x_grid)/100 + 1
y_m = np.array(y_grid)/100 + 12

ax.scatter(x_m, y_m,s=100)
#ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue')
for i, txt in enumerate(electrode_id_list):
    ax.annotate(txt, (x_m[i]+0.1, y_m[i]),fontsize=16)
plt.xlabel('Distance [m]', fontsize=18, fontweight='bold')
plt.ylabel ('Distance [m]', fontsize=20, fontweight='bold')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
x_section = np.arange(-10, 600)
x_shadow1 = np.arange(-20, x_coords[8])
x_shadow2 = np.arange(-20, x_coords[5])
x_shadow3 = np.arange(-20, x_coords[4])

# test
x_p = [506.66313421812913]
y_p = [-1118.6067180325601]

y = -1162.6763900644542
x = 486.11310865727773

point = [300, -1022.2381158883322]

line1 = line_equation(x_section, sector_m1, sector_b1)
line2 = line_equation(x_section, sector_m2, sector_b2)
line3 = line_equation(x_section, sector_m3, sector_b3)

shadow11 = line_equation(x_shadow1, shadow1_m1, shadow1_b1)
shadow12 = line_equation(x_shadow1, shadow1_m2, shadow1_b2)

shadow21 = line_equation(x_shadow2, shadow2_m1, shadow2_b1)
shadow22 = line_equation(x_shadow2, shadow2_m2, shadow2_b2)

shadow31 = line_equation(x_shadow3, shadow3_m1, shadow3_b1)
shadow32 = line_equation(x_shadow3, shadow3_m2, shadow3_b2)

#ax.plot(x_section, line1)
#ax.plot(x_section, line2)
#ax.plot(x_section, line3)
#ax.plot(x_shadow1, shadow11, color='y')
#ax.plot(x_shadow1, shadow12, color='y')
#ax.plot(x_shadow2, shadow21, color='g')
#ax.plot(x_shadow2, shadow22, color='g')
#ax.plot(x_shadow3, shadow31, color='black')
#ax.plot(x_shadow3, shadow32, color='black')

#ax.plot(x, y, 'o', color='b')
#ax.plot(x_p, y_p, 'o', color='r')
#ax.plot(point[0], point[1], 'o', color='green')
plt.xlim(-2, 11)
plt.ylim(-2,14)
plt.gca().set_aspect('equal', adjustable='box')
fig.set_size_inches(10,18)
plt.savefig('measure_grid_max-master.png',dpi=300)
plt.show()
