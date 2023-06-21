import numpy
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from grid_middle_line_calculator import grid_middle_line_calculator

def line_equation(x,slope,intercept):
    return slope*x + intercept

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
# ist 55 von und 55 y

Grid_master_right_data= [70, 30],[70, 35], [65, 45],[55, 50],[50, 50],[60, 45],[55, 55]
degree_master_left=[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
degree_master_right=[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]

# calculate distance in master grid between elektrodes on the right row
end_diff_master_left =40

master_distance_left =[]

for i in range(len(Grid_master_right_data)-1):
    a=[Grid_master_right_data[i][1],Grid_master_right_data[i+1][0]]
    master_distance_left.append(a)
end_numbers = [Grid_master_right_data[-1][1], end_diff_master_left]
master_distance_left.append(end_numbers)
#Distancase slave grid, [0] dsitance and degree between slave and master

slave_distance_grid_data_left =[50,50],[50,50],[50,50],[65,35],[25,75],[25,75],[25,75],[25,75]
degree_slave_left = [30,30],[30,30],[30,30],[30,65],[65,65],[65,65],[65,65],[65,65]

slave_distance_grid_data_right = [70,35],[50,50],[50,50],[65,35],[75,25],[75,25],[75,25],[75,25]
degree_slave_right = [0,30], [30,30], [30,30], [30,65], [65,65], [65,65], [65,65],[65,65]


left_dist_list = [master_distance_left, slave_distance_grid_data_left]
left_dist_list = np.concatenate(left_dist_list)

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

#### calculating y coordinates based on yd ( degree of y point orientation from point befor) and yl ( point distance)
#### for right and left side of grid
for i in range(len(left_dist_list)):
    yd1 = np.cos(np.deg2rad(left_degree_list[i][0]))
    yl1 = np.int64(left_dist_list[i][0])
    a = yl1 *yd1
    yd2 = np.cos(np.deg2rad(left_degree_list[i][1]))
    yl2 = np.int64(left_dist_list[i][1])
    b = yl2 * yd2
    yl = y_coordinates_left[-1] - a -b
    if isinstance(yl,np.ndarray) is True:
        yl = yl[0]

    y_coordinates_left.extend([yl])

    yds1 = np.cos(np.deg2rad(left_degree_list[i][0]))
    yr1 = np.int64(left_dist_list[i][0])
    bb = yr1 *yd1
    yds2 = np.cos(np.deg2rad(left_degree_list[i][1]))
    yr2 = np.int64(left_dist_list[i][1])
    aa = yr2 *yds2
    yr = y_coordinates_right[-1] - aa - bb
    if isinstance(yr, np.ndarray) is True:
        yr = yr[0]
    y_coordinates_right.extend([yr])
## caluclationg x coordinates by distance and xd ( degree difference from pojt befor)
for i in range(len(left_dist_list)):

    xd1 = np.sin(np.deg2rad(left_degree_list[i][0]))
    x1 = np.int64(left_dist_list[i][0])*xd1

    xd2 = np.sin(np.deg2rad(left_degree_list[i][1]))
    x2 = np.int64(left_dist_list[i][1])*xd2

    xl = x_coordinates_left[-1] +x1 +x2
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

x_grid= []
x_grid.extend(x_coordinates_left)
x_grid.extend(x_coordinates_right)

y_grid= []
y_grid.extend(y_coordinates_left)
y_grid.extend(y_coordinates_right)


electrode_id_list = [16, 15, 14, 13, 12, 11, 10, 9, 32, 31, 30, 29, 28, 27, 26, 25, 8, 7, 6, 5, 4, 3, 2, 1, 24, 23, 22, 21, 20, 19, 18, 17]
electrode_arrangement_list = [25, 17, 26, 18, 27, 19, 28, 20, 29, 21, 30, 22, 31, 23, 32, 24, 9, 1, 10, 2, 11, 3, 12, 4,
                              13, 5, 14, 6, 15, 7, 16, 8]


np.save('elecrode_pos_list', electrode_id_list)
np.save('elecrode_arrangement',electrode_arrangement_list)
np.save('x_coordinates_grid', x_grid)
np.save('y_coordinates_grid', y_grid)

# get middle grid
middle_line_grid = grid_middle_line_calculator(x_grid,y_grid,electrode_id_list,electrode_arrangement_list)
x_coords = [point[0] for point in middle_line_grid]
y_coords = [point[1] for point in middle_line_grid]


electrode_id_list = np.array(electrode_id_list)

#calculate sector

x11 = x_grid[np.where(electrode_id_list == 9)[0][0]]
x12 = x_grid[np.where(electrode_id_list == 1)[0][0]]
y11= y_grid[np.where(electrode_id_list == 9)[0][0]]
y12= y_grid[np.where(electrode_id_list == 1)[0][0]]

sector_m1 = ((y12-y11)/(x12-x11))
sector_b1 = y11 - sector_m1*x11

#bewteen l2:l3'
x21 = x_grid[np.where(electrode_id_list == 22)[0][0]]
x22 = x_grid[np.where(electrode_id_list == 30)[0][0]]
y21 = y_grid[np.where(electrode_id_list == 22)[0][0]]
y22 = y_grid[np.where(electrode_id_list == 30)[0][0]]
sector_m2 = ((y22-y21)/(x22-x21))
sector_b2 = y21 - sector_m2*x21

        #bewteen l3:l4
x31 = x_grid[np.where(electrode_id_list == 21)[0][0]]
x32 = x_grid[np.where(electrode_id_list == 29)[0][0]]
y31 = y_grid[np.where(electrode_id_list == 21)[0][0]]
y32 = y_grid[np.where(electrode_id_list == 29)[0][0]]
sector_m3 =((y32-y31)/(x32-x31))
sector_b3 = y31 - sector_m3 * x31

#plot
fig,ax = plt.subplots()
ax.scatter(x_grid, y_grid)
ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue')
for i, txt in enumerate(electrode_id_list):
    ax.annotate(txt, (x_grid[i],y_grid[i]))

x_section = range(-10,600)

#test
x_p = [506.66313421812913]
y_p = [-1118.6067180325601]


y = -1162.6763900644542
x = 486.11310865727773

point = [300, -1022.2381158883322]





line1 = line_equation(x_section,sector_m1,sector_b1)
line2 = line_equation(x_section,sector_m2,sector_b2)
line3 = line_equation(x_section,sector_m3,sector_b3)

ax.plot(x_section,line1)
ax.plot(x_section,line2)
ax.plot(x_section,line3)
ax.plot(x, y, 'o', color='b')
ax.plot(x_p, y_p, 'o', color='r')
ax.plot(point[0],point[1], 'o', color='green')

plt.xlim(-50, 750)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('measure_grid_max-master.png')
plt.show()
