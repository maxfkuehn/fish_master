from datetime import datetime

a = ['10:56:50', '10:57:00', '10:57:10', '10:57:20', '10:57:30', '10:57:40', '10:57:50', '10:58:00', '10:58:10', '10:58:20', '10:58:30', '10:58:40', '10:58:50', '10:59:00', '10:59:10', '10:59:20']

datetime_list = []

for time_str in a:
    dt = datetime.strptime(time_str, '%H:%M:%S')
    datetime_list.append(dt)

print(datetime_list)
