
import numpy as np
import glob
import os
from IPython import embed
import pandas as pd
import math
import matplotlib.pyplot as plt

#### thunderloader
#laods .csv 
#
#
# 
def thunderloader(path_way):
    # load all *waveeod files
    laoded_names_waveeod = []
    os.chdir(path_way)
    # go thourgh all waveeodf.csv files and get the file names and sort them 
    for file in glob.glob('*waveeodfs.csv'):
        laoded_names_waveeod.append(file)

    file_names_waveeod = sorted(laoded_names_waveeod)
    
    # loop over every waveeodf.csv and load data of the right fish

    combined_waveforms_mean_all_fish = []
    combined_waveforms_times_all_fish = []

    combined_waveforms_mean_single_fish = []
    combined_waveforms_time_single_fish = []

    normalised_waveform_mean_af = []
    normalised_waveform_mean_sf = []

    counter = 0
    previous_name = 'first'

    amount_fish_present_list = []
    fish_present = []
    
    
    #list of which window of then second recording it belons to
    t_window_list = []
    t_window = []
    af_frequency_list = []

    for name in file_names_waveeod:
        previous_name.split('_')
        counter += 1
        
        if  previous_name == 'first':

            index_raw = name.split('_')
            fish_eodf = int(index_raw[4].split('-')[0])
            #laod fish index
            waveeod_data = pd.read_csv(path_way+'/'+name)

            # look in the files where the EODf of the fish is clsoest to the one in the title, get the index and load EODf waveforms of the fish
            waveod_index = np.where(waveeod_data['EODf'] == min(waveeod_data['EODf'], key=lambda x:abs(x-fish_eodf)))[0][0]
            csv_index = str(waveeod_data['index'][waveod_index])
    
            #cheking how many fish are present in the recording
            amount_fish_present = len(waveeod_data['index'])
           
            

            if csv_index.isnumeric() == False:
                continue
            
            if  name.split('_')[3] == '1':
                if name.split('-')[1] == 't0s':
                    tw = 10
                elif name.split('-')[1] == 't2s':
                    tw = 12                
                elif name.split('-')[1] == 't4s':
                    tw = 14
                elif name.split('-')[1] == 't6s':
                    tw = 16
                else:
                    tw = 18
            elif  name.split('_')[3] == '2':
                if name.split('-')[1] == 't0s':
                    tw = 20
                elif name.split('-')[1] == 't2s':
                    tw = 22                
                elif name.split('-')[1] == 't4s':
                    tw = 24
                elif name.split('-')[1] == 't6s':
                    tw = 26
                else:
                    tw = 28                   
            else:
                if name.split('-')[1] == 't0s':
                    tw = 30
                elif name.split('-')[1] == 't2s':
                    tw = 32                
                elif name.split('-')[1] == 't4s':
                    tw = 34
                elif name.split('-')[1] == 't6s':
                    tw = 36
                else:
                    tw = 38


           
            # open waveform data
            path_wafeform_csv = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-eodwaveform-'+ csv_index+'.csv'

            eod_waveform_loaded = pd.read_csv(path_wafeform_csv)


             # read data with wavefish specific attributes of selected fish


            wavefish_csv_path = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-wavefish.csv'
            wavefish_attributes = pd.read_csv(wavefish_csv_path)
         
            rel_peak_amp_sf = wavefish_attributes['relpeakampl'][int(csv_index)]
            ptp_dist = wavefish_attributes['p-p-distance'][int(csv_index)]
            minptp_dist = wavefish_attributes['min-p-p-distance'][int(csv_index)]
            trough_width = wavefish_attributes['troughwidth'][int(csv_index)]  
            thd_sf = wavefish_attributes['thd'][int(csv_index)]
 

            # laod wave time and amplitudeint(
            time_wf =   np.array(eod_waveform_loaded['time'])
            mean_wf = np.array(eod_waveform_loaded['mean'])

            combined_waveforms_mean_single_fish.append(mean_wf)
            combined_waveforms_time_single_fish.append(time_wf)
            normalised_waveform_mean_sf.append((np.array(mean_wf)-min(mean_wf))/np.max(mean_wf-min(mean_wf)))
            fish_present.append(amount_fish_present)
            
            t_window.append(tw)
            
            previous_name = name
            if counter == len(file_names_waveeod):
                combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                af_frequency_list.append(fish_eodf)
                amount_fish_present_list.append(fish_present)
                t_window_list.append(t_window)
                normalised_waveform_mean_af.append(normalised_waveform_mean_sf)
        else:
            pn = previous_name.split('_')
            nn = name.split('_')

            if pn[1] == nn[1] and pn[2] == nn[2]:
                        
                index_raw = name.split('_')
                fish_eodf = int(index_raw[4].split('-')[0])

                waveeod_data = pd.read_csv(path_way+'/'+name)
                # look in the files where the EODf of the fish is clsoest to the one in the title, get the index and load EODf waveforms of the fish
                waveod_index = np.where(waveeod_data['EODf'] == min(waveeod_data['EODf'], key=lambda x:abs(x-fish_eodf)))[0][0]
                csv_index = str(waveeod_data['index'][waveod_index])

                if csv_index.isnumeric() == False:
                    continue
                


                path_wafeform_csv = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-eodwaveform-'+ csv_index+'.csv'
              
                eod_waveform_loaded = pd.read_csv(path_wafeform_csv)

                #cheking how many fish are present in the recording
                amount_fish_present = len(waveeod_data['index'])
                #check which time window
                if  name.split('_')[3] == '1':
                    if name.split('-')[1] == 't0s':
                        tw = 10
                    elif name.split('-')[1] == 't2s':
                        tw = 12                
                    elif name.split('-')[1] == 't4s':
                        tw = 14
                    elif name.split('-')[1] == 't6s':
                        tw = 16
                    else:
                        tw = 18
                elif  name.split('_')[3] == '2':
                    if name.split('-')[1] == 't0s':
                        tw = 20
                    elif name.split('-')[1] == 't2s':
                        tw = 22                
                    elif name.split('-')[1] == 't4s':
                        tw = 24
                    elif name.split('-')[1] == 't6s':
                        tw = 26
                    else:
                        tw = 28                   
                else:
                    if name.split('-')[1] == 't0s':
                        tw = 30
                    elif name.split('-')[1] == 't2s':
                        tw = 32                
                    elif name.split('-')[1] == 't4s':
                        tw = 34
                    elif name.split('-')[1] == 't6s':
                        tw = 36
                    else:
                        tw = 38


                time_wf =   np.array(eod_waveform_loaded['time'])
                mean_wf = np.array(eod_waveform_loaded['mean'])

                #get waveform time mean and standardised mean
            
      
                combined_waveforms_mean_single_fish.append(mean_wf)
                combined_waveforms_time_single_fish.append(time_wf)
                normalised_waveform_mean_sf.append((np.array(mean_wf)-min(mean_wf))/np.max(mean_wf-min(mean_wf)))
                fish_present.append(amount_fish_present)
                
                t_window.append(tw)
                    

                previous_name = name
                if counter == len(file_names_waveeod):
                    combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                    combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                    af_frequency_list.append(fish_eodf)
                    amount_fish_present_list.append(fish_present)
                    t_window_list.append(t_window)
                    normalised_waveform_mean_af.append(normalised_waveform_mean_sf)
            
            else:
               
                combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                af_frequency_list.append(fish_eodf)
                amount_fish_present_list.append(fish_present)
                t_window_list.append(t_window)
                normalised_waveform_mean_af.append(normalised_waveform_mean_sf)

                combined_waveforms_mean_single_fish = []
                combined_waveforms_time_single_fish = []
                fish_present= []
                t_window = []
                normalised_waveform_mean_sf = []

                index_raw = name.split('_')
                fish_eodf = int(index_raw[4].split('-')[0])

                waveeod_data = pd.read_csv(path_way+'/'+name)
                # look in the files where the EODf of the fish is clsoest to the one in the title, get the index and load EODf waveforms of the fish
                waveod_index = np.where(waveeod_data['EODf'] == min(waveeod_data['EODf'], key=lambda x:abs(x-fish_eodf)))[0][0]
                csv_index = str(waveeod_data['index'][waveod_index])

                if csv_index.isnumeric() == False:
                    continue

                path_wafeform_csv = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-eodwaveform-'+ csv_index+'.csv'
                                          
                eod_waveform_loaded = pd.read_csv(path_wafeform_csv)

                # read data with wavefish specific attributes of selected fish


                wavefish_csv_path = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-wavefish.csv'
                wavefish_attributes = pd.read_csv(wavefish_csv_path)
            
                rel_peak_amp_sf = wavefish_attributes['relpeakampl'][int(csv_index)]
                ptp_dist = wavefish_attributes['p-p-distance'][int(csv_index)]
                minptp_dist = wavefish_attributes['min-p-p-distance'][int(csv_index)]
                trough_width = wavefish_attributes['troughwidth'][int(csv_index)]  
                thd_sf = wavefish_attributes['thd'][int(csv_index)]
    
                #cheking how many fish are present in the recording
                amount_fish_present = len(waveeod_data['index'])
                #check which time window
                if  name.split('_')[3] == '1':
                    if name.split('-')[1] == 't0s':
                        tw = 10
                    elif name.split('-')[1] == 't2s':
                        tw = 12                
                    elif name.split('-')[1] == 't4s':
                        tw = 14
                    elif name.split('-')[1] == 't6s':
                        tw = 16
                    else:
                        tw = 18
                elif  name.split('_')[3] == '2':
                    if name.split('-')[1] == 't0s':
                        tw = 20
                    elif name.split('-')[1] == 't2s':
                        tw = 22                
                    elif name.split('-')[1] == 't4s':
                        tw = 24
                    elif name.split('-')[1] == 't6s':
                        tw = 26
                    else:
                        tw = 28                   
                else:
                    if name.split('-')[1] == 't0s':
                        tw = 30
                    elif name.split('-')[1] == 't2s':
                        tw = 32                
                    elif name.split('-')[1] == 't4s':
                        tw = 34
                    elif name.split('-')[1] == 't6s':
                        tw = 36
                    else:
                        tw = 38
                        
                time_wf =   np.array(eod_waveform_loaded['time'])
                mean_wf = np.array(eod_waveform_loaded['mean'])

      
                combined_waveforms_mean_single_fish.append(mean_wf)
                combined_waveforms_time_single_fish.append(time_wf)
                normalised_waveform_mean_sf.append((np.array(mean_wf)-min(mean_wf))/np.max(mean_wf-min(mean_wf)))
                fish_present.append(amount_fish_present)

                t_window.append(tw)
                    

                previous_name = name
                if counter == len(file_names_waveeod):
                    combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                    combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                    af_frequency_list.append(fish_eodf)
                    amount_fish_present_list.append(fish_present)
                    t_window_list.append(t_window)
                    normalised_waveform_mean_af.append(normalised_waveform_mean_sf)
    
    for a in range(len(combined_waveforms_mean_all_fish)):
        window_array = np.array(t_window_list[a])
        
        idx_1 = np.where(window_array < 20)[0]
        idx_2 = np.where((window_array < 30) & (window_array > 18))[0]
        idx_3 = np.where(window_array > 28)[0]
        fig,((ax11,ax12,ax13,ax14),(ax21,ax22,ax23,ax24)) = plt.subplots(2,4)    

        for ix1 in idx_1:
            ax11.plot(combined_waveforms_times_all_fish[a][ix1],combined_waveforms_mean_all_fish[a][ix1])
            ax21.plot(combined_waveforms_times_all_fish[a][ix1],normalised_waveform_mean_af[a][ix1]) 
            ax11.title.set_text('#1 means')
            ax21.title.set_text('#1 normalized')

        for ix2 in idx_2:
            ax12.plot(combined_waveforms_times_all_fish[a][ix2],combined_waveforms_mean_all_fish[a][ix2]) 
            ax22.plot(combined_waveforms_times_all_fish[a][ix2],normalised_waveform_mean_af[a][ix2])
            ax12.title.set_text('#2 means')
            ax22.title.set_text('#2 normalized')           
            

        for ix3 in idx_3:
            ax13.plot(combined_waveforms_times_all_fish[a][ix3],combined_waveforms_mean_all_fish[a][ix3]) 
            ax23.plot(combined_waveforms_times_all_fish[a][ix3],normalised_waveform_mean_af[a][ix3]) 
            ax13.title.set_text('#3 means')
            ax23.title.set_text('#3 normalized')

        for i in range(len(combined_waveforms_mean_all_fish[a])):
            ax14.plot(combined_waveforms_times_all_fish[a][i],combined_waveforms_mean_all_fish[a][i]) 
            ax24.plot(combined_waveforms_times_all_fish[a][i],normalised_waveform_mean_af[a][i])     
            ax14.title.set_text('All Combined means')
            ax14.title.set_text('All Combined means normalized')
        fig.suptitle(str(af_frequency_list[a]))

        

        single_waveforms,((ax10,ax11,ax12,ax13,ax14),(ax20,ax21,ax22,ax23,ax24),(ax30,ax31,ax32,ax33,ax34)) = plt.subplots(3,5)

        ax1_array = np.array([ax10,ax11,ax12,ax13,ax14])
        ax2_array = np.array([ax20,ax21,ax22,ax23,ax24])
        ax3_array = np.array([ax30,ax31,ax32,ax33,ax34])

    
        for i1 in range(len(idx_1)):
            inx = idx_1[i1]
            ax1_array[i1].title.set_text('N = %d' % amount_fish_present_list[a][inx])
            ax1_array[i1].plot(combined_waveforms_times_all_fish[a][inx],combined_waveforms_mean_all_fish[a][inx])
        for i2 in range(len(idx_2)):
            imx = idx_2[i2]
            ax2_array[i2].plot(combined_waveforms_times_all_fish[a][imx],combined_waveforms_mean_all_fish[a][imx])
            ax2_array[i2].title.set_text('N = %d' % amount_fish_present_list[a][imx])
        for i3 in range(len(idx_3)):
            iox = idx_3[i3]
            ax3_array[i3].plot(combined_waveforms_times_all_fish[a][iox],combined_waveforms_mean_all_fish[a][iox])
            ax3_array[i3].title.set_text('N = %d' % amount_fish_present_list[a][iox])

        ax10.set_ylabel('#1')

        ax20.set_ylabel('#2')

        ax30.set_ylabel('#3')



        single_waveforms.suptitle(str(af_frequency_list[a])+': all single windows normalised')
        single_waveforms.supxlabel('Time []')
        single_waveforms.supylabel('Mean Amplitude')
        
    
    


##### Main function

if __name__ == "__main__":
    ### setting

    path_wavforms = '/home/kuehn/data/waveforms/thunderfish'
    time_window_thunderfish = ['t0','t2','t4','t6','t8']
    thunderloader(path_wavforms)

