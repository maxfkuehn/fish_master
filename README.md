# Scripts from my Neuro Bio Master

Hello, this is a short README, which highlights the most important function of my Master thesis and Hiwi.
Most scripts have hard coded pathways for the data of my thesis.
To be usbale in another context or other data sets, pathways needs to be adjusted in the scripts.
Should have made them functions which work with Input. I am aware and sorry for my sins of the past!
Also i want to beg for forgiveness for my naming crimes, my script and function names are horrible.

1. # Fuse_data_master_slave.py

  This script fuses data from a master and slave recording grid. Data used must be preprocessed by
  Till Raabs detection algorythm (Name? wurde mit Hand die Punkte in der gleichen Spur einer ID zu geordnen)
  Scripts need the folowing data:

  '/ident.npy' list of IDs. Every point that corresponds to one EOD track in one channel.
  '/freq.npy' Frequency of each data point.
  '/timeidx.npy'  Time index of each data point.
  '/sign.npy' Amplitude of each data point.
  '/times.npy' Time of each data point.

2. # Connect_fish_id_grid.py

  This sctipt creates matching pairs (mapa) of IDs and channel, that belong to one fish.
  It checks if two tracks in different channel have the same frequency and are overlapping in time with a       
  tolerance of 60s. IDs that have the same frequency range and overlapp in time are paired.
  To check the frequency, i took the median frequency of both tracks and compared them. If the differnce was         with in 2 Hz, they were considered the same.
  Mapa: [Ch x ,Id x][Ch y ,Id y]

  TLDR: Connect tracks over different channels and saves them as ID Ch pairs and arrange them to one fish.
  
3. # create_fish_list.py

  Takes the mapa and creates a fish list. 
  The fish list contains a list of fish, which each contains two list. One List of all unique ids and the     
  correspondinf channels in which these ids track where recorded.
  So fish_list[0][0] is the first id and corresponding channel that belong to fish0 , 
  Fish_list[0][0][:] -> all channels of first fish
  Fish_list[0][1][:] -> all ids of first fish

  The fish list is the main data structure where fish data is managed and loaded.
  
4. # doppelganger_hunter_fl-py

  The name name says it all. There is the possiblity for duplicted id/ch pairs in the fl.
  This checks for duplicates and deletes them if found.

5. # 




6. 
   



   
  
  
  

  
