
import numpy as np
import glob
import os
from IPython import embed
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import mplcursors
import sys
import matplotlib.cm as cm
from random import randrange
from thunderfish.eodanalysis import fourier_series
from scipy.optimize import curve_fit
from typing import List

#### thunderloader

# 
def thunderloader(path_way):
    '''
    Loads waveform and waveform attribudes of every thunderfish csv in a given folder

    Input: 
    path_way: Pathway to the folder with csv data
    
    Returns:
    Each returned variable contains a list of fish which contains a list of each loaded attribute.
    combined_waveforms_mean_all_fish: standardised amplitudes of each recording (bewteen 0:1)
    combined_waveforms_times_all_fish: timepoints of each recording
    ppamplitude_af: Peak-to-peak amplitude of the extracted waveform in the units of the input data
    ppdistance_af: Time between peak and trough relative to EOD period
    peakwidth_af: Width of the peak at the averaged amplitude relative to EOD period
    thd_af:  Total harmonic distortion i.e. square root of sum of amplitudes squared of harmonics relative to amplitude of fundamental
    leftpeak_af: Time from negative zero crossing to trough relative to EOD period.
    rightpeak_af: Time from trough to positive zero crossing relative to EOD period 

    '''

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
    
    ''' wave fish element empty lists'''

    ppamplitude_af = []
    thd_af = []
    peakwidth_af = []
    leftpeak_af = []
    rightpeak_af = []
    ppdistance_af= []
    min_width_af = []
    min_pp_dist_af = []
    rel_peak_amp_af = []


    ppamplitude_sf = []
    thd_sf = []
    peakwidth_sf = []
    leftpeak_sf = []
    rightpeak_sf = []
    ppdistance_sf= []
    min_width_sf = []
    min_pp_dist_sf = []
    rel_peak_amp_sf = []


    
    #list of which window of then second recording it belons af
    
    af_frequency_list = []

    for name in file_names_waveeod:

        previous_name.split('_')
        print(name,end='\r')
        counter += 1
        
        if  previous_name == 'first':

            index_raw = name.split('_')
            fish_eodf = int(index_raw[4].split('-')[0])

            waveeod_data = pd.read_csv(path_way+'/'+name)
            # look in the files where the EODf of the fish is clsoest to the one in the title, get the index and load EODf waveforms of the fish
            waveod_index = np.where(waveeod_data['EODf'] == min(waveeod_data['EODf'], key=lambda x:abs(x-fish_eodf)))[0][0]
            csv_index = str(waveeod_data['index'][waveod_index])

            #cheking how many fish are present in the recording
            amount_fish_present = len(waveeod_data['index'])

            path_wafeform_csv = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-eodwaveform-'+ csv_index+'.csv'

            eod_waveform_loaded = pd.read_csv(path_wafeform_csv)

            time_wf =   np.array(eod_waveform_loaded['time'])
            mean_wf = np.array(eod_waveform_loaded['mean'])

            combined_waveforms_mean_single_fish.append(mean_wf)
            combined_waveforms_time_single_fish.append(time_wf)
            norm_mean = mean_wf/max(mean_wf)
            normalised_waveform_mean_sf.append(norm_mean)
            
            fish_present.append(amount_fish_present)

            '''
            load wavefish data of every recording to compare wave parameters
            opens wavefish.csv of thunderfish and extracts data
            '''

            path_wavefish_csv = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-wavefish.csv'
    
            loaded_wavefish_csv = pd.read_csv(path_wavefish_csv)
      
            ppamplitude_sr = np.array(loaded_wavefish_csv['p-p-amplitude'][int(csv_index)])
            thd_sr = np.array(loaded_wavefish_csv['thd'][int(csv_index)])
            peakwidth_sr = np.array(loaded_wavefish_csv['peakwidth'][int(csv_index)])
            leftpeak_sr = np.array(loaded_wavefish_csv['leftpeak'][int(csv_index)])
            rightpeak_sr = np.array(loaded_wavefish_csv['rightpeak'][int(csv_index)])
            ppdistance_sr= np.array(loaded_wavefish_csv['p-p-distance'][int(csv_index)])
            min_width_sr = np.array(loaded_wavefish_csv['minwidth'][int(csv_index)])
            min_pp_dist_sr = np.array(loaded_wavefish_csv['min-p-p-distance'][int(csv_index)])
            rel_peak_amp_sr = np.array(loaded_wavefish_csv['relpeakampl'][int(csv_index)])

            ppamplitude_sf .append(ppamplitude_sr)
            thd_sf .append(thd_sr)
            peakwidth_sf .append(peakwidth_sr)
            leftpeak_sf .append(leftpeak_sr)
            rightpeak_sf .append(rightpeak_sr)
            ppdistance_sf.append(ppdistance_sr)
            min_width_sf.append(min_width_sr)         
            min_pp_dist_sf.append(min_pp_dist_sr)
            rel_peak_amp_sf.append(rel_peak_amp_sr)


            previous_name = name

            if counter == len(file_names_waveeod):
                combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                af_frequency_list.append(fish_eodf)
                amount_fish_present_list.append(fish_present)
    
                normalised_waveform_mean_af.append(normalised_waveform_mean_sf)

                ppamplitude_af .append(ppamplitude_sf)
                thd_af .append(thd_sf)
                peakwidth_af .append(peakwidth_sf)
                leftpeak_af .append(leftpeak_sf)
                rightpeak_af .append(rightpeak_sf)
                ppdistance_af.append(ppdistance_sf)
                min_width_af.append(min_width_sf)
                min_pp_dist_af.append(min_pp_dist_sf)
                rel_peak_amp_af.append(rel_peak_amp_sf)
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

                time_wf =   np.array(eod_waveform_loaded['time'])
                mean_wf = np.array(eod_waveform_loaded['mean'])

                #get waveform time mean and standardised mean
            
      
                combined_waveforms_mean_single_fish.append(mean_wf)
                combined_waveforms_time_single_fish.append(time_wf)
                normalised_waveform_mean_sf.append(mean_wf/max(mean_wf))
                fish_present.append(amount_fish_present)
                
                '''
                load wavefish data of every recording to compare wave parameters
                opens wavefish.csv of thunderfish and extracts data
                '''


                path_wavefish_csv = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-wavefish.csv'
        
                loaded_wavefish_csv = pd.read_csv(path_wavefish_csv)

                ppamplitude_sr = np.array(loaded_wavefish_csv['p-p-amplitude'][int(csv_index)])
                thd_sr = np.array(loaded_wavefish_csv['thd'][int(csv_index)])
                peakwidth_sr = np.array(loaded_wavefish_csv['peakwidth'][int(csv_index)])
                leftpeak_sr = np.array(loaded_wavefish_csv['leftpeak'][int(csv_index)])
                rightpeak_sr = np.array(loaded_wavefish_csv['rightpeak'][int(csv_index)])
                ppdistance_sr= np.array(loaded_wavefish_csv['p-p-distance'][int(csv_index)])
                min_width_sr = np.array(loaded_wavefish_csv['minwidth'][int(csv_index)])
                min_pp_dist_sr = np.array(loaded_wavefish_csv['min-p-p-distance'][int(csv_index)])
                rel_peak_amp_sr = np.array(loaded_wavefish_csv['relpeakampl'][int(csv_index)])

                ppamplitude_sf .append(ppamplitude_sr)
                thd_sf .append(thd_sr)
                peakwidth_sf .append(peakwidth_sr)
                leftpeak_sf .append(leftpeak_sr)
                rightpeak_sf .append(rightpeak_sr)
                ppdistance_sf.append(ppdistance_sr)
                min_width_sf.append(min_width_sr)         
                min_pp_dist_sf.append(min_pp_dist_sr)
                rel_peak_amp_sf.append(rel_peak_amp_sr)
                        

                    

                previous_name = name
                if counter == len(file_names_waveeod):
                    combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                    combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                    af_frequency_list.append(fish_eodf)
                    amount_fish_present_list.append(fish_present)
                    normalised_waveform_mean_af.append(normalised_waveform_mean_sf)
                    ppamplitude_af .append(ppamplitude_sf)
                    thd_af .append(thd_sf)
                    peakwidth_af .append(peakwidth_sf)
                    leftpeak_af .append(leftpeak_sf)
                    rightpeak_af .append(rightpeak_sf)
                    ppdistance_af.append(ppdistance_sf)
                    min_width_af.append(min_width_sf)
                    min_pp_dist_af.append(min_pp_dist_sf)
                    rel_peak_amp_af.append(rel_peak_amp_sf)
            
            else:
               
                combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                af_frequency_list.append(fish_eodf)
                amount_fish_present_list.append(fish_present)
                normalised_waveform_mean_af.append(normalised_waveform_mean_sf)

                ppamplitude_af .append(ppamplitude_sf)
                thd_af .append(thd_sf)
                peakwidth_af .append(peakwidth_sf)
                leftpeak_af .append(leftpeak_sf)
                rightpeak_af .append(rightpeak_sf)
                ppdistance_af.append(ppdistance_sf)
                min_width_af.append(min_width_sf)
                min_pp_dist_af.append(min_pp_dist_sf)
                rel_peak_amp_af.append(rel_peak_amp_sf)


                ppamplitude_sf = []
                thd_sf = []
                peakwidth_sf = []
                leftpeak_sf = []
                rightpeak_sf = []
                ppdistance_sf= []
                min_width_sf = []
                min_pp_dist_sf = []
                rel_peak_amp_sf = []

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
                #cheking how many fish are present in the recording
                amount_fish_present = len(waveeod_data['index'])

                        
                time_wf =   np.array(eod_waveform_loaded['time'])
                mean_wf = np.array(eod_waveform_loaded['mean'])

      
                combined_waveforms_mean_single_fish.append(mean_wf)
                combined_waveforms_time_single_fish.append(time_wf)
                normalised_waveform_mean_sf.append(mean_wf/np.max(mean_wf))
                fish_present.append(amount_fish_present)
                '''
                load wavefish data of every recording to compare wave parameters
                opens wavefish.csv of thunderfish and extracts data
                '''

                path_wavefish_csv = path_way+ '/' + name.split('-')[0]+'-'+name.split('-')[1]+'-wavefish.csv'
        
                loaded_wavefish_csv = pd.read_csv(path_wavefish_csv)

                ppamplitude_sr = np.array(loaded_wavefish_csv['p-p-amplitude'][int(csv_index)])
                thd_sr = np.array(loaded_wavefish_csv['thd'][int(csv_index)])
                peakwidth_sr = np.array(loaded_wavefish_csv['peakwidth'][int(csv_index)])
                leftpeak_sr = np.array(loaded_wavefish_csv['leftpeak'][int(csv_index)])
                rightpeak_sr = np.array(loaded_wavefish_csv['rightpeak'][int(csv_index)])
                ppdistance_sr= np.array(loaded_wavefish_csv['p-p-distance'][int(csv_index)])
                min_width_sr = np.array(loaded_wavefish_csv['minwidth'][int(csv_index)])
                min_pp_dist_sr = np.array(loaded_wavefish_csv['min-p-p-distance'][int(csv_index)])
                rel_peak_amp_sr = np.array(loaded_wavefish_csv['relpeakampl'][int(csv_index)])

                ppamplitude_sf .append(ppamplitude_sr)
                thd_sf .append(thd_sr)
                peakwidth_sf .append(peakwidth_sr)
                leftpeak_sf .append(leftpeak_sr)
                rightpeak_sf .append(rightpeak_sr)
                ppdistance_sf.append(ppdistance_sr)
                min_width_sf.append(min_width_sr)         
                min_pp_dist_sf.append(min_pp_dist_sr)
                rel_peak_amp_sf.append(rel_peak_amp_sr)
                    

                previous_name = name
                if counter == len(file_names_waveeod):
                    combined_waveforms_mean_all_fish.append(combined_waveforms_mean_single_fish)
                    combined_waveforms_times_all_fish.append(combined_waveforms_time_single_fish)
                    af_frequency_list.append(fish_eodf)
                    amount_fish_present_list.append(fish_present)
                    
                    normalised_waveform_mean_af.append(normalised_waveform_mean_sf)
                    ppamplitude_af .append(ppamplitude_sf)
                    thd_af .append(thd_sf)
                    peakwidth_af .append(peakwidth_sf)
                    leftpeak_af .append(leftpeak_sf)
                    rightpeak_af .append(rightpeak_sf)
                    ppdistance_af.append(ppdistance_sf)
                    min_width_af.append(min_width_sf)
                    min_pp_dist_af.append(min_pp_dist_sf)
                    rel_peak_amp_af.append(rel_peak_amp_sf)
    
    return combined_waveforms_mean_all_fish,normalised_waveform_mean_af, combined_waveforms_times_all_fish, min_pp_dist_af, min_width_af, thd_af, leftpeak_af, rightpeak_af,rel_peak_amp_af, af_frequency_list

def best_cor_fish(cor_matrix,cor_fn):
    '''
    Function that calculates the n# highest correlatied waveforms that are most representive for the average/majority of waveform of recordings

    input:
    cor_matrix: correlation matrix of each recording compared to other recordings of a fish, can be list of list (multiples fishs eacha  list with recordings cor_matrix)
    cor_fn: Number n# of best recordings returned by function 
    '''  

    best_cof_rec_af =[]

    for cormax in cor_matrix:
        try:
            if cormax.any():
                if len(cormax)<cor_fn:
                    best_cof_rec_af.append(np.arange(1,len(cormax)))
                    continue

                cor_average_rec = []

                for coma in cormax:
                    average_cof = np.average(coma)
                    cor_average_rec.append(average_cof)
                    
                max_average_cof = np.where(cor_average_rec == np.max(cor_average_rec))[0][0].tolist()

                best_rec = cormax[max_average_cof]

                idx_av_cof = np.arange(1,len(best_rec)) 

                zip_av_cor = zip(best_rec,idx_av_cof)

                sort_zip = sorted(zip_av_cor, key=lambda x: x[1])[-cor_fn:-1]
                
                av_max,idx_max = zip(*sort_zip)

                idx_max = [x.tolist() for x in idx_max]

                best_rec_sf = [max_average_cof]
                best_rec_sf.extend(idx_max)

                best_cof_rec_af.append(best_rec_sf)
                
        except AttributeError:
            best_cof_rec_af.append([])
    
    return(best_cof_rec_af)



def thunder_cof_matrix(combined_waveforms_mean_all_fish, combined_waveforms_times_all_fish,normalised_waveform_mean_af):
    '''
    Function to calculate the cof matrix of waveforms of each recordings of one fish

    input:
    combined_waveforms_mean_all_fish: mean average amplitude, list of fish, each list contains an array or list for each recording with mean amplitudes of this individual fish
    combined_waveforms_times_all_fish: timepoints, list of fish, each list contains an array or list for each recording with mean amplitudes of this individual fish
    normalised_waveform_mean_af: standartised mean amplitude, list of fish, each list contains an array or list for each recording with mean amplitudes of this individual fish

    returns:

    cor_matrix_af :  via list of fish, each containing the cof matrix of each recording in relation to all other recodings of the same individual fish
    cor_matrix_unique: list of fish containing array whitch unique combination of corelation pairs
    cor_pair_matrix; list of fish containing an array of every unique pair of recordings 
    '''     

    int_waveforms_af = []
    unique_cor_af = []
    cor_matrix_af =[]
    cor_pair_matrix = []
    
    for af in range(len(combined_waveforms_mean_all_fish)):
        int_waveforms_sf = []

        sf_nor_waveform = normalised_waveform_mean_af[af]
        sf_time = combined_waveforms_times_all_fish[af]
        
        min_sf_time = []
        max_sf_time = []

        if sf_time == [] or len(sf_time) < 2:
            cor_matrix_af.append([])
            continue

        for sft in sf_time:
            min_sf_time.append(min(sft))
            max_sf_time.append(max(sft))

        start_time = max(min_sf_time)
        end_time = min(max_sf_time)
        
        int_time = np.arange(start_time, end_time, 0.03)

        for sf in range(len(sf_nor_waveform)):
            try:
                intpol = interpolate.interp1d(sf_time[sf],sf_nor_waveform[sf], kind='cubic' , fill_value="extrapolate")
                ynew = intpol(int_time)
                int_waveforms_sf.append(ynew)

            except ValueError:
                print('Interpolation Error')
                continue

        int_waveforms_af.append(int_waveforms_sf)

        """
        Create Correlation Matrix
        """
        
        cor_matrix = np.corrcoef(int_waveforms_sf)

        pair_matrix = []
        for f in range(len(cor_matrix)):
            pair = []
            for icf in range(len(cor_matrix[f])):
                pair.append([f, icf])

            pair_matrix.append(pair)


        # get unique combinations of pairs
        try:
            triu_mat = np.triu(cor_matrix,1)
        except TypeError:
            print('Value Error 494')
         

        triu_array = triu_mat.ravel() #unfold list
        unique_cor = triu_array[np.where(triu_array!=0)[0]] #get unique pairs from unfold list as new array
        unique_cor_af.append(unique_cor)
        pair_array = np.array([x for xs in pair_matrix for x in xs])# unfold pair matrix and get inique pairs with index from correlation triu_mat
        unique_pairs = pair_array[np.where(triu_array!=0)[0]]
        cor_pair_matrix.append(unique_pairs)
    
        
        min_cor = min(unique_cor)
        max_cor =  max(unique_cor)
        average_cor = np.mean(unique_cor)
        std_cor = np.std(unique_cor)
        cor_matrix_af.append(cor_matrix)


    return cor_matrix_af,unique_cor_af, cor_pair_matrix


def waveform_period_normaliser(amplitude,time,freq = None):
    '''
    Function that takes amplitude, time and optionally frequency  of a periodic signal and returns two period long wvaeforms normalised in time .
    Periodstart point is  centered around the peak of the first period.

    Input: 
    amplitude: list of amplitude values of recorded periodic signal  
    time: list of Timepoints of the recorded amplitude values fore eachr ecording
    freq: Optional, list of Frequencies of periodic signals, if not given freq get calculated (not finished yet)

    output: 
    norm:amplitude: list of new amplitude arrays starting at maximum and ending after 2 periods
    norm_time: list of normalised timepoints also cut and timed same way as amplitude
    not_useable_fish_index: index of fish which was not useable due to too short timewindow or wrong input parameters
    '''
    print('Normalizing data')

    #fourier expoansion to calculate wafeform over bigger time window
 
    fex_time_af = []
    fex_amp_af = []
    empty_counter = 0
    empty_index = []
    # number of periods that thr foureier extansion creates
    number_periods = 6
    
    for f in range(len(time)):
        t_period = round(1/freq[f]*1000,2) #calculated time of one period in ms rounded to second decimale
        xtime = np.arange(time[f][0],time[f][0]+number_periods*t_period,0.01) #time array with length of 3 periods of waveform in 0,01 ms steps
        try:
            fex_amp = fourier_expansion(time[f],amplitude[f],freq[f]/1000,xtime) #use fourier expansion from thunderfish to get wafeform over xtime time frame
        except RuntimeError:
            print(f"Recording {f} unusable due recording not including a full period of EODf waveform")
            fex_time_af.append([])
            fex_amp_af.append([])
            empty_counter +=1
            empty_index.append(f)
            continue
        

        # normalise time by multiplying with frequency and divide by ms 
        norm_t = xtime*(freq[f]/1000)
        norm_amp = fex_amp/max(fex_amp)
        #if fex_amp > 1.1 there is a error in calulating fourier due to wrong given freq or amplitude, recording cant be used 
        if max(fex_amp) >1.1:
            print(f"Wrong calculated amplitude due to wrong frequency, recording {f} not usable")
            fex_time_af.append([])
            fex_amp_af.append([])
            empty_counter +=1
            empty_index.append(f)
            
        else:
            fex_time_af.append(norm_t)
            fex_amp_af.append(norm_amp)

    
    # detect first peak and start recording at peak and set start to 0
    
    peak_all_recordings = []
    counter_peak = -1
    time_return_list = []
    amp_return_list = []
    for amp in fex_amp_af:

        counter_peak += 1
        try:
            peak,_ = find_peaks(amp)
            
            if len(peak) == 0:
                peak_all_recordings.append([])
                empty_index.append(counter_peak)

            elif len(peak) <= number_periods:
                peak_all_recordings.append(peak)
            # if to many peaks are detected to fit to the period number,
            # peaks are checked and only valid if they are either at a apropriate differnce to next point
            # or if not have a higher amplitude 
            else:
                real_peak = []
                for p in peak:
                    if amp[p]> 0.9:
                        real_peak.append(p)
                
                jumpcounter = 0 # binär counter that checks if next iterations can be left out
                checked_rp = []

                for i in range(len(real_peak)):
                    
                    if jumpcounter == 1: # leaves out itteration but checks if next iteration can also be left out
                        if i == len(real_peak)-1:
                            continue
                        elif time_p[real_peak[i+1]]-time_p[real_peak[i]] > 0.3:
                            jumpcounter = 0
                            continue
                        else:
                            if real_peak[i]-real_peak[i+1] > 0:
                                jumpcounter = 0
                                continue
                            else:
                                continue


                    if i == 0:
                        time_p = fex_time_af[counter_peak]
                        if time_p[real_peak[i+1]]-time_p[real_peak[i]] > 0.3:
                            checked_rp.append(real_peak[i])
            
                        else:
                            if real_peak[i]-real_peak[i+1] > 0:
                                checked_rp.append(real_peak[i])
                                jumpcounter = 1
                            else:
                                continue

                    elif i == len(real_peak)-1:

                        time_p = fex_time_af[counter_peak]
                        if time_p[real_peak[i]]-time_p[real_peak[i-1]] > 0.3:
                            checked_rp.append(real_peak[i])
            
                        else:
                            if real_peak[i]-real_peak[i-1] > 0:
                                checked_rp.append(real_peak[i])
                    else:
                        right_time_df = time_p[real_peak[i+1]]-time_p[real_peak[i]] 
                        left_time_df  = time_p[real_peak[i]]-time_p[real_peak[i-1]]

                        right_amp_df = amp[real_peak[i]]-amp[real_peak[i+1]]
                        left_amp_df  = amp[real_peak[i]]-amp[real_peak[i-1]]

                        if right_time_df <= 0.3:
                            if right_amp_df > 0:
                                checked_rp.append(real_peak[i])
                                jumpcounter = 1
                        
                            else:
                                continue
                        else:
                            checked_rp.append(real_peak[i])
       
                peak_all_recordings.append(checked_rp)
            
        except TypeError:
            peak_all_recordings.append([])
            empty_index.append(counter_peak)

    empty_index = np.unique(empty_index)
    #set new peak at t=0 and change time array accordingly also normalize period
    fig,(ax1,ax2) = plt.subplots(1,2)
    for i in range(len(peak_all_recordings)):

        if i in empty_index:
            time_return_list.append([])
            amp_return_list.append([])
            continue 
        
        else:
            amp_list = fex_amp_af[i]
        
            if amp_list[0]>0:
                try:
                    pk = peak_all_recordings[i][1]
                    third_peak = peak_all_recordings[i][3]  
                except IndexError:
                    print('Second Peak index error')
            else:
                nextindex = 1
                for par in peak_all_recordings[i]:
                    pk = par
                    if len(peak_all_recordings) > 1:
                        try:
                            third_peak = peak_all_recordings[i][nextindex+1]
                        except IndexError:
                            print('Second peak index error')
                    
                    if amp_list[pk] > 0.8:
                        break
                    nextindex += 1

        # set first peaik to 0 and modify timepoints so that second peak is at 1 (exacttly one period) 
        new_t = fex_time_af[i]-fex_time_af[i][pk]
        fix_ratio = 2/new_t[third_peak]
        fixed_time = new_t*fix_ratio

        ax1.title.set_text('Uncorrected')
        ax1.plot(new_t,amp_list)
        ax2.title.set_text('Corrected by time ratio')
        ax2.plot(fixed_time,amp_list)
 
        time_return = new_t[pk:third_peak]
        amp_return = amp_list[pk:third_peak]
        
        amp_return_list.append(amp_return)
        time_return_list.append(time_return)



        #new_t= zeroed_t*(1/normalised_time[i][second_peak])
    
    plt.close()
    plt.show()
    #interpolate data so they have same lengh after overlaing waveforms
    time_return = []
    amp_return = []
    time_interpolation = np.linspace(0,2,400) # new time frame for interpolation


    final_fig = plt.figure()
    normed_list = []
    for idx in range(len(time_return_list)):
        #normalise data again to guarante end time of 2

        try:
            time_interpolation = np.linspace(0,2,400) # new time frame for interpolation
            normed_to_two = time_return_list[idx]*(2/time_return_list[idx][-1])
            normed_list.append(normed_to_two[-1])
            interpid_amp = interpolate.interp1d(normed_to_two,amp_return_list[idx],kind='cubic',fill_value="extrapolate")(time_interpolation)
            amp_return.append(interpid_amp)
            time_return.append(time_interpolation)

            plt.plot(time_interpolation,interpid_amp)
            

        except IndexError:
            amp_return.append([])
            time_return.append([])
            continue

    plt.close()
    plt.show()

    print('Normalization finished')
    return time_return,amp_return,empty_index




# fourier expansion out of thundefish to calculate waveform extansion to a given timerange and time resolution
'''

Input:
time: time array of data
eod: measured eod amp array
eodf: frequency of eod
xtime: array of time range and resolution of the output data    

'''
def fourier_expansion(time, eod, eodf, xtime):
    ampl = 0.5*(np.max(eod)-np.min(eod))
    n_harm = 20
    params = [eodf]
    for i in range(1, n_harm+1):
        params.extend([ampl/i, 0.0])
    popt, pcov = curve_fit(fourier_series, time, eod, params)
    return fourier_series(xtime, *popt)


def interactive_cluster_plot(X_pca,timepoint,amplitude,kmeans,show=True):
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6)) 

    scatter = ax_left.scatter(X_pca[:, 0], X_pca[:, 1], #plot pca data on left plot
                                c=kmeans, alpha=0.5,picker=True, pickradius = 1)
    
    ax_left.set_title('PCA of stand_amp with KMeans Clusters')
    ax_left.set_xlabel('Principal Component 1')
    ax_left.set_ylabel('Principal Component 2')
    # Adding a colorbar to show the cluster labels
    plt.colorbar(scatter, ax=ax_left, label='Kmeans Cluster Labels')

    line, = ax_right.plot([], [], color='orange')  #create empty placeholder plot

        # Funktion zum Aktualisieren des rechten Plots
    def update_right_plot(index):
        x_data = timepoint[index]
        y_data = amplitude[index]
        line.set_data(x_data, y_data)
        ax_right.relim()
        ax_right.autoscale_view()
        fig.canvas.draw()                     

    # Function to handle clicks on the scatter plot
    def on_click(event):
        # Check if the mouse click was inside the axes
        if event.inaxes == ax_left:
    
            # Get the data from the scatter plot
            xdata = scatter.get_offsets()[:, 0]
            ydata = scatter.get_offsets()[:, 1]

            # Find the index of the clicked point
            distances = np.sqrt((xdata - event.xdata) ** 2 + (ydata - event.ydata) ** 2)
            index = np.argmin(distances)  # Get the index of the closest point

            # Highlight the selected point
            scatter.set_sizes([20 if i != index else 200 for i in range(len(xdata))])  # Change size of clicked point

            # Update the right plot with the new index
            update_right_plot(index)

    # Connect the click event to the on_click function
    fig.canvas.mpl_connect('button_press_event', on_click)

    if show==True:
        plt.show() 
    return fig
    plt.tight_layout()  # Adjust layout to avoid overlap  

# function to cluster attributes of thunderfish csv data

def thunderclusterer(path_way_csv, data_type = 'parameters' ,mode =  None):
    '''
    Put waveforms of recordings of each fish into categories of a given csv data set create by thunderfish.  
    This is achieved by k-clustering either waveform or waveform parameters.

    Input:
    path_way: Pathway to the folder with csv data 
    *data_type: data_type = 'waveform', 'parameter'  defualt = 'parameter' , waveform conmpares waveform, parameter waveform parameters
    *mode: mode = 'best', 'None' can choose to only use best corelated recording of each fish to cluster, default uses all recordings

    returns:
    Plot of k clustering and plot of waveforms of each fish with cluster kategory in title 

    '''
    #number of waveforms with highest cor per fish that count
    cor_fn = 4
    # print modes used
    print('Mode: '+ mode + ', datatype: ' + data_type)
    # use thunderloader to get extracted .csv data
    
    amp,stand_amp, timepoints, min_pp_dist, minwidth, tdh, leftpeak, rightpeak,relpeakamp, af_freq = thunderloader(path_wavforms)
    
    # changeable paramaters

    min_amp = 0.0004  # minimal amplitude to be valid data point


    #create sorted pandas dtata frame

    data_cluster = {'min_pp_dist':min_pp_dist,'relpeakamp':relpeakamp,'minwidth':minwidth,'tdh':tdh,'leftpeak':leftpeak ,'rightpeak':rightpeak,'freq':af_freq}
    df_cluster_sorted = pd.DataFrame(data_cluster)
    
    # create a list of the corresponding fish_nr for every data point
    fish_nr_list  = [] 
    af_freq_list = []

    for fish_nr in df_cluster_sorted.index:
        fish_nr_list.append(np.ones(len(df_cluster_sorted['tdh'][fish_nr]),dtype=int)*fish_nr)
        af_freq_list.append(np.ones(len(df_cluster_sorted['tdh'][fish_nr]),dtype=int)*df_cluster_sorted['freq'].iloc[fish_nr])
    
    # create pandas frame of data but unfolded
    flat_stamp = [x for xs in stand_amp for x in xs]
    flat_time = [x for xs in timepoints for x in xs]
    flat_amp = [x for xs in amp for x in xs]
    #make all time windows start at 0 s
   
    for ft in range(len(flat_time)) :
        ft_list = flat_time[ft]
        transfer_value = min(ft_list)*-1
        flat_time[ft] = ft_list+transfer_value
   
    data_cluster_unfolded = {'amp':flat_amp,'stand_amp':flat_stamp,'timepoints':flat_time,'min_pp_dist':np.concatenate(min_pp_dist),'relpeakamp':np.concatenate(relpeakamp),'minwidth':np.concatenate(minwidth),'tdh':np.concatenate(tdh),'leftpeak':np.concatenate(leftpeak) ,'rightpeak':np.concatenate(rightpeak),'fish':np.concatenate(fish_nr_list),'freq':np.concatenate(af_freq_list)}
    
    df_cluster_unfolded = pd.DataFrame(data_cluster_unfolded) 
    
    #check if fish fall below min_amp theshold and save them in delete list:

    max_ap_af_f = []



    for ap_af in amp:
        max_ap_sf = []
        for ap_sf in ap_af:
            map = np.max(ap_sf)
            max_ap_sf.append(map)
        max_ap_af_f.append(max_ap_sf)
 
    del_list_amp = []
    
    for idx in range(len(np.concatenate(max_ap_af_f))):
    
        if np.concatenate(max_ap_af_f)[idx] < min_amp:
            
            del_list_amp.append(idx)    
            
    df_cluster_unfolded.drop(del_list_amp, inplace=True)
    df_cluster_unfolded.reset_index(inplace=True)
    fishnr = df_cluster_unfolded['fish']

    

    min_pp_dist_corrected = []
    relpeakamp_corrected = []
    minwidth_corrected = []
    tdh_corrected = []
    leftpeak_corrected = []
    rightpeak_corrected = []
    stand_amp_corrected = []
    timepoints_corrected = []
    amp_corrected = []
    freq_corrected = []

    for fish in range(np.max(fishnr)):
        idx_fish = np.where(fishnr == fish)[0]
        minppd = []
        relpam = []
        tdh_ = []
        minwd = []
        lpeak = []
        rpeak = []
        stamp = []
        tp = []
        am = []
        frq = []

        for i in idx_fish:
            
            minppd.append(df_cluster_unfolded['min_pp_dist'].iloc[i]) 
            relpam.append(df_cluster_unfolded['relpeakamp'].iloc[i])
            minwd.append(df_cluster_unfolded['minwidth'].iloc[i])
            tdh_.append(df_cluster_unfolded['tdh'].iloc[i])
            lpeak.append(df_cluster_unfolded['leftpeak'].iloc[i])
            rpeak.append(df_cluster_unfolded['rightpeak'].iloc[i])
            stamp.append(df_cluster_unfolded['stand_amp'].iloc[i])
            tp.append(df_cluster_unfolded['timepoints'].iloc[i])
            am.append(df_cluster_unfolded['amp'].iloc[i])     
            frq.append(df_cluster_unfolded['freq'].iloc[i])

        min_pp_dist_corrected.append(minppd)
        relpeakamp_corrected.append(relpam)
        minwidth_corrected.append(minwd)
        tdh_corrected.append(tdh_)
        leftpeak_corrected.append(lpeak)
        rightpeak_corrected.append(rpeak)
        stand_amp_corrected.append(stamp)
        timepoints_corrected.append(tp)
        amp_corrected.append(am)
        freq_corrected.append(frq)
    
    corrected = {'stand_amp':stand_amp_corrected,'timepoints':timepoints_corrected,'min_pp_dist':min_pp_dist_corrected,'relpeakamp':relpeakamp_corrected,'minwidth':minwidth_corrected,'tdh':tdh_corrected,'leftpeak':leftpeak_corrected ,'rightpeak':rightpeak_corrected,}
    df_cluster_sorted = pd.DataFrame(corrected)

    if mode == 'best':
        #load correlation of each fish waveforms compared to other recordings of the same fish
        cor_matrix,cor_matrix_unique,cor_pair = thunder_cof_matrix(amp_corrected,timepoints_corrected,stand_amp_corrected)
        # find highest correlations pairs of waveforms of each fish that are representative to the most common wave type (highest average of the  sum of correlations og single recording compared to all other channels)

        best_fitting_rec = best_cor_fish(cor_matrix,cor_fn)

        df_best_cluster = pd.DataFrame()


        for fish in range(len(best_fitting_rec)):
            idx_fish = np.where(df_cluster_unfolded['fish']==fish)[0]
            for i in best_fitting_rec[fish]:
                bidx = idx_fish[i]
                df_best_cluster= pd.concat([df_best_cluster,df_cluster_unfolded.loc[[bidx]]],ignore_index=True)    
            

        if data_type=='parameters':
            scaler = StandardScaler()
            df_best_cluster[['min_pp_dist_n','relpeakamp_n','minwidth_n','tdh_n','leftpeak_n','rightpeak_n']] = scaler.fit_transform(df_best_cluster[['min_pp_dist','relpeakamp','minwidth','tdh','leftpeak','rightpeak']])
            

            # clustering 

            kmeans = KMeans(n_clusters=3)
            kmeans.fit(df_best_cluster[['min_pp_dist_n','relpeakamp','minwidth_n','tdh_n','leftpeak_n','rightpeak_n']])
            df_best_cluster['kmeans_2']=kmeans.labels_

         
            #plot cluster
            cluster_figure,((ax00,ax01,ax02,ax03,ax04),(ax10,ax11,ax12,ax13,ax14),(ax20,ax21,ax22,ax23,ax24)) = plt.subplots(3,5)

            ax00.scatter(x=df_best_cluster['min_pp_dist'],y=df_best_cluster['minwidth'],c=df_best_cluster['kmeans_2'])
            ax00.title.set_text(' ')
            ax00.set_xlabel('min_pp_dist')
            ax00.set_ylabel('peakwidth')

            ax01.scatter(x=df_best_cluster['min_pp_dist'],y=df_best_cluster['tdh'],c=df_best_cluster['kmeans_2'])
            ax01.title.set_text(' ')
            ax01.set_xlabel('min_pp_dist')
            ax01.set_ylabel('tdh')

            ax02.scatter(x=df_best_cluster['min_pp_dist'],y=df_best_cluster['leftpeak'],c=df_best_cluster['kmeans_2'])
            ax02.title.set_text(' ')
            ax02.set_xlabel('min_pp_dist')
            ax02.set_ylabel('leftpeak_n')

            ax03.scatter(x=df_best_cluster['min_pp_dist'],y=df_best_cluster['relpeakamp'],c=df_best_cluster['kmeans_2'])
            ax03.title.set_text(' ')
            ax03.set_xlabel('min_pp_dist')
            ax03.set_ylabel('relpeakamp')

            ax04.scatter(x=df_best_cluster['min_pp_dist'],y=df_best_cluster['rightpeak'],c=df_best_cluster['kmeans_2'])
            ax04.title.set_text(' ')
            ax04.set_xlabel('min_pp_dist')
            ax04.set_ylabel('rightpeak')

            ax10.scatter(x=df_best_cluster['relpeakamp'],y=df_best_cluster['minwidth'],c=df_best_cluster['kmeans_2'])
            ax10.title.set_text(' ')
            ax10.set_xlabel('relpeakamp')
            ax10.set_ylabel('minwidth')


            ax11.scatter(x=df_best_cluster['relpeakamp'],y=df_best_cluster['tdh'],c=df_best_cluster['kmeans_2'])
            ax11.title.set_text(' ')
            ax11.set_xlabel('relpeakamp')
            ax11.set_ylabel('tdh')

            ax12.scatter(x=df_best_cluster['relpeakamp'],y=df_best_cluster['leftpeak'],c=df_best_cluster['kmeans_2'])
            ax12.title.set_text(' ')
            ax12.set_xlabel('relpeakamp')
            ax12.set_ylabel('leftpeak')


            ax13.scatter(x=df_best_cluster['relpeakamp'],y=df_best_cluster['rightpeak'],c=df_best_cluster['kmeans_2'])
            ax13.title.set_text(' ')
            ax13.set_xlabel('relpeakamp')
            ax13.set_ylabel('rightpeak')

            ax14.scatter(x=df_best_cluster['minwidth'],y=df_best_cluster['tdh'],c=df_best_cluster['kmeans_2'])
            ax14.title.set_text(' ')
            ax14.set_xlabel('minwidth')
            ax14.set_ylabel('tdh')

            ax20.scatter(x=df_best_cluster['minwidth'],y=df_best_cluster['leftpeak'],c=df_best_cluster['kmeans_2'])
            ax20.title.set_text(' ')
            ax20.set_xlabel('peakwidth')
            ax20.set_ylabel('leftpeak_n')

            ax21.scatter(x=df_best_cluster['minwidth'],y=df_best_cluster['rightpeak'],c=df_best_cluster['kmeans_2'])
            ax21.title.set_text(' ')
            ax21.set_xlabel('minwidth')
            ax21.set_ylabel('rightpeak')

            ax22.scatter(x=df_best_cluster['tdh'],y=df_best_cluster['leftpeak'],c=df_best_cluster['kmeans_2'])
            ax22.title.set_text(' ')
            ax22.set_xlabel('tdh')
            ax22.set_ylabel('leftpeak_n')

            ax23.scatter(x=df_best_cluster['tdh'],y=df_best_cluster['minwidth'],c=df_best_cluster['kmeans_2'])
            ax23.title.set_text(' ')
            ax23.set_xlabel('tdh')
            ax23.set_ylabel('rightpeak')    

            ax24.scatter(x=df_best_cluster['leftpeak'],y=df_best_cluster['rightpeak'],c=df_best_cluster['kmeans_2'])
            ax24.title.set_text(' ')
            ax24.set_xlabel('leftpeak')
            ax24.set_ylabel('rightpeak')    


            plt.show()
            plt.close()
            

            # for each fish plot data points and cluster for each recording 

            for a in range(df_best_cluster['fish'].iloc[-1]):
                
                wave_clust, (ax0,ax1,ax2,ax3) = plt.subplots(4,1)

                ax1_array = np.array([ax0,ax1,ax2,ax3])
                
                idx_1 = np.where(df_best_cluster['fish']==a)[0]

                #for fish in range(664):

                for i1 in range(len(idx_1)):
                    ix1 = idx_1[i1]
                    ax1_array[i1].title.set_text('fish '+ str(df_best_cluster['fish'].iloc[ix1])+': cluster ' + str(df_best_cluster['kmeans_2'].iloc[ix1]))
                    ax1_array[i1].plot(df_best_cluster['timepoints'].iloc[ix1],df_best_cluster['stand_amp'].iloc[ix1])
               
                frqfish = df_best_cluster['freq'].iloc[idx_1[0]] 

                wave_clust.suptitle(f'Frequence: {frqfish} Hz')
                wave_clust.supxlabel('Time')
                wave_clust.supylabel('Amplitude')
                plt.show()



        elif data_type=='waveform':
            
            #get overall max and min time of all recordings

            min_time =[]
            max_time = []

            for f in range(len(df_best_cluster['stand_amp'])):
                sf_tp = df_best_cluster['timepoints'][f]

                min_t = min(sf_tp)
                max_t = max(sf_tp)

                min_time.append(min_t)
                max_time.append(max_t)

            start_time = max(min_time)
            end_time = min(max_time)
            
            #interpolate data to equal size
            int_waveforms = []
            int_times = []
            for af in range(len(df_best_cluster['stand_amp'])):
                
                sf_nor_waveform = df_best_cluster['stand_amp'][af]
                sf_time = df_best_cluster['timepoints'][af]
                
                int_time = np.arange(start_time, end_time, 0.03)
            
                try:
                    intpol = interpolate.interp1d(sf_time,sf_nor_waveform, kind='cubic',fill_value="extrapolate" )
                    ynew = intpol(int_time)
                    int_waveforms.append(ynew)
                    int_times.append(int_time)
                except ValueError:
                        print('Interpolation Error detected')
                        continue
            

            cluster_time,cluster_amp,empty_indx = waveform_period_normaliser(int_waveforms,int_times,df_best_cluster['freq'])

            # add new time
            df_best_cluster['stand_amp'] = cluster_amp
            #add new amp
            df_best_cluster['timepoints'] = cluster_time
            #delete enot usable data
            df_best_cluster.drop(empty_indx,inplace=True)

            # k-cluster waveform
            amp_used_cluster = df_best_cluster['stand_amp'].to_list()

            kmeans = KMeans(n_clusters=5)
            kmeans.fit(amp_used_cluster)
            df_best_cluster['kmeans_3']=kmeans.labels_
            # to get 'außreißer'out we kick those out of data
            # Drop rows where kmeans_3 hast the shortest list (pur 2 bad data points)

            count = df_best_cluster['kmeans_3'].value_counts()
            bad_data = count.idxmin()
            df_best_cluster = df_best_cluster[df_best_cluster['kmeans_3'] != bad_data]
            amp_used_cluster = df_best_cluster['stand_amp'].to_list()
            

            # PCA of data to see which amplitude at time x was most impectfull
            pca = PCA(n_components=10)  # We want to reduce to 2 components for plotting
            X_pca = pca.fit_transform(amp_used_cluster)

            #cluster again with cleaned data
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(X_pca)
            df_best_cluster['kmeans_3']=kmeans.labels_

             #cluster again with cleaned data
            kmeans = KMeans(n_clusters=4)
            kmeans.fit(X_pca)
            df_best_cluster['kmeans_4']=kmeans.labels_

            #cluster again with cleaned data
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(X_pca)
            df_best_cluster['kmeans_5']=kmeans.labels_

            

            # Plotting the scatter with KMeans labels
            EODf = df_best_cluster['freq'].to_list()
            plot_amp=df_best_cluster['stand_amp'].to_list()
            plot_time = df_best_cluster['timepoints'].to_list()
            kmeans_label = df_best_cluster['kmeans_3'].to_list()
            
            stand_freq = (np.array(EODf)-min(EODf))/(max(EODf)-min(EODf))
    
            fused_data = np.column_stack((plot_amp,stand_freq))

            k_means_3f,X_pca_f = kmeans_including_freq(plot_amp,EODf,k_kmeans=3)
            k_means_4f,X_pca_f  = kmeans_including_freq(plot_amp,EODf,k_kmeans=4)
            k_means_5f,X_pca_f  = kmeans_including_freq(plot_amp,EODf,k_kmeans=5)

            kmeans_label4  = df_best_cluster['kmeans_4']
            kmeans_label5  = df_best_cluster['kmeans_5']

            embed()

            kcluster_best_n(fused_data)

            fig1 = interactive_cluster_plot(X_pca=X_pca,timepoint=plot_time,amplitude=plot_amp,kmeans=kmeans_label,show=False) 
            a1 = fig1.axes[0] 
            a1.set_title('k3 No freq')           
            fig2 = interactive_cluster_plot(X_pca_f,plot_time,plot_amp,k_means_3f,show=False)
            a2 = fig2.axes[0]
            a2.set_title('k3 With Freq')
            fig3 = interactive_cluster_plot(X_pca=X_pca,timepoint=plot_time,amplitude=plot_amp,kmeans=kmeans_label4,show=False) 
            a3 = fig3.axes[0] 
            a3.set_title('k4 No freq')           
            fig4 = interactive_cluster_plot(X_pca_f,plot_time,plot_amp,k_means_4f,show=False)
            a4 = fig4.axes[0]
            a4.set_title('k4 With Freq')
            fig5 = interactive_cluster_plot(X_pca=X_pca,timepoint=plot_time,amplitude=plot_amp,kmeans=kmeans_label5,show=False) 
            a5 = fig5.axes[0] 
            a5.set_title('k5 No freq')           
            fig6 = interactive_cluster_plot(X_pca_f,plot_time,plot_amp,k_means_5f,show=False)
            a6 = fig6.axes[0]
            a6.set_title('k4 With Freq')

            plt.show()
            embed()
            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            plt.title('PCA of stand_amp')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')

            plt.show()

            idx_1 = range(4)
            idx_2 = range(4,8)
            idx_3 = range(8,12)
            idx_4 = range(12,16)
    

            test_picture, ((ax00,ax01,ax02,ax03),(ax10,ax11,ax12,ax13),(ax20,ax21,ax22,ax23),(ax30,ax31,ax32,ax33)) = plt.subplots(4,4)

            ax1_array = np.array([ax00,ax01,ax02,ax03])
            ax2_array = np.array([ax10,ax11,ax12,ax13])
            ax3_array = np.array([ax20,ax21,ax22,ax23])
            ax4_array = np.array([ax30,ax31,ax32,ax33])

            
            #for fish in range(664):

            for i1 in range(len(idx_1)):
                ix1 = idx_1[i1]
                ax1_array[i1].title.set_text('frq '+ str(df_best_cluster['freq'].iloc[ix1])+':cluster ' + str(df_best_cluster['kmeans_3'].iloc[ix1]))
                ax1_array[i1].plot(df_best_cluster['timepoints'].iloc[ix1],df_best_cluster['stand_amp'].iloc[ix1])
            for i2 in range(len(idx_2)):
                ix2 = idx_2[i2]
                ax2_array[i2].title.set_text('frq '+str(df_best_cluster['freq'].iloc[ix2])+': cluster ' +str(df_best_cluster['kmeans_3'].iloc[ix2]))
                ax2_array[i2].plot(df_best_cluster['timepoints'].iloc[ix2],df_best_cluster['stand_amp'].iloc[ix2])
            for i3 in range(len(idx_3)):
                ix3 = idx_3[i3]
                ax3_array[i3].title.set_text('frq '+str(df_best_cluster['freq'].iloc[ix3])+': cluster ' +str(df_best_cluster['kmeans_3'].iloc[ix3]))
                ax3_array[i3].plot(df_best_cluster['timepoints'].iloc[ix3],df_best_cluster['stand_amp'].iloc[ix3])
            for i4 in range(len(idx_4)):
                ix4 = idx_4[i4]
                ax4_array[i4].title.set_text('frq '+str(df_best_cluster['freq'].iloc[ix4])+': cluster ' +str(df_best_cluster['kmeans_3'].iloc[ix4]))
                ax4_array[i4].plot(df_best_cluster['timepoints'].iloc[ix4],df_best_cluster['stand_amp'].iloc[ix4])                                  
            
            #plt.show()
            plt.close()
            
            """         for i in range(df_best_cluster['fish'].iloc[-1]):
                idx_1 = range(4*i,4*i+4)
                singlefishplot, (axx00,axx01,axx02,axx03) = plt.subplots(1,4)
                ax_array = np.array([axx00,axx01,axx02,axx03])

                for z in range(len(idx_1)):
                    
                    ix = idx_1[z]
                    ax_array[z].title.set_text('frq '+ str(df_best_cluster['freq'].iloc[ix])+':cluster ' + str(df_best_cluster['kmeans_3'].iloc[ix]))
                    ax_array[z].plot(df_best_cluster['timepoints'].iloc[ix],df_best_cluster['stand_amp'].iloc[ix])
                
                #plt.show()
            """    
            return df_best_cluster # return the pandas dataframe
        


    else: 

        corrected = {'stand_amp':stand_amp_corrected,'timepoints':timepoints_corrected,'min_pp_dist':min_pp_dist_corrected,'relpeakamp':relpeakamp_corrected,'minwidth':minwidth_corrected,'tdh':tdh_corrected,'leftpeak':leftpeak_corrected ,'rightpeak':rightpeak_corrected,}
        df_cluster_sorted = pd.DataFrame(corrected)

        if data_type == 'parameters':
        
            #create corrected DataFrame with lists of lists

        


            # get max amplitude for each fish to later delete low amplitude data points
            #create sorted pandas dtata frame

            # delete low amp 
            #df_cluster_unfolded=df_cluster_unfolded.drop(df_cluster_unfolded[df_cluster_unfolded['max_amp']<min_amp].index)
            
            # standardise data points for l clustering, standardies x = (x^n - mean)/standart deviation
            scaler = StandardScaler()
            df_cluster_unfolded[['min_pp_dist_n','relpeakamp_n','minwidth_n','tdh_n','leftpeak_n','rightpeak_n']] = scaler.fit_transform(df_cluster_unfolded[['min_pp_dist','relpeakamp','minwidth','tdh','leftpeak','rightpeak']])
            

            # clustering 

            kmeans = KMeans(n_clusters=3)
            kmeans.fit(df_cluster_unfolded[['min_pp_dist_n','relpeakamp','minwidth_n','tdh_n','leftpeak_n','rightpeak_n']])
            df_cluster_unfolded['kmeans_2']=kmeans.labels_




            # add cluseting to sorted df

            counter = 0
            kmeans_list = []

            for fi in df_cluster_sorted.index:
                end_indx = counter + len(df_cluster_sorted['tdh'][fi])
                kmeans_sorted = df_cluster_unfolded['kmeans_2'][counter:end_indx]
                kmeans_list.append(kmeans_sorted)


            idx_new_colums = len(df_cluster_sorted.columns)
            df_cluster_sorted.insert(idx_new_colums,'kmeans_2',kmeans_list)

            #plot cluster
            cluster_figure,((ax00,ax01,ax02,ax03,ax04),(ax10,ax11,ax12,ax13,ax14),(ax20,ax21,ax22,ax23,ax24)) = plt.subplots(3,5)

            ax00.scatter(x=df_cluster_unfolded['min_pp_dist'],y=df_cluster_unfolded['minwidth'],c=df_cluster_unfolded['kmeans_2'])
            ax00.title.set_text(' ')
            ax00.set_xlabel('min_pp_dist')
            ax00.set_ylabel('peakwidth')

            ax01.scatter(x=df_cluster_unfolded['min_pp_dist'],y=df_cluster_unfolded['tdh'],c=df_cluster_unfolded['kmeans_2'])
            ax01.title.set_text(' ')
            ax01.set_xlabel('min_pp_dist')
            ax01.set_ylabel('tdh')

            ax02.scatter(x=df_cluster_unfolded['min_pp_dist'],y=df_cluster_unfolded['leftpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax02.title.set_text(' ')
            ax02.set_xlabel('min_pp_dist')
            ax02.set_ylabel('leftpeak_n')

            ax03.scatter(x=df_cluster_unfolded['min_pp_dist'],y=df_cluster_unfolded['relpeakamp'],c=df_cluster_unfolded['kmeans_2'])
            ax03.title.set_text(' ')
            ax03.set_xlabel('min_pp_dist')
            ax03.set_ylabel('relpeakamp')

            ax04.scatter(x=df_cluster_unfolded['min_pp_dist'],y=df_cluster_unfolded['rightpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax04.title.set_text(' ')
            ax04.set_xlabel('min_pp_dist')
            ax04.set_ylabel('rightpeak')

            ax10.scatter(x=df_cluster_unfolded['relpeakamp'],y=df_cluster_unfolded['minwidth'],c=df_cluster_unfolded['kmeans_2'])
            ax10.title.set_text(' ')
            ax10.set_xlabel('relpeakamp')
            ax10.set_ylabel('minwidth')


            ax11.scatter(x=df_cluster_unfolded['relpeakamp'],y=df_cluster_unfolded['tdh'],c=df_cluster_unfolded['kmeans_2'])
            ax11.title.set_text(' ')
            ax11.set_xlabel('relpeakamp')
            ax11.set_ylabel('tdh')

            ax12.scatter(x=df_cluster_unfolded['relpeakamp'],y=df_cluster_unfolded['leftpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax12.title.set_text(' ')
            ax12.set_xlabel('relpeakamp')
            ax12.set_ylabel('leftpeak')


            ax13.scatter(x=df_cluster_unfolded['relpeakamp'],y=df_cluster_unfolded['rightpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax13.title.set_text(' ')
            ax13.set_xlabel('relpeakamp')
            ax13.set_ylabel('rightpeak')

            ax14.scatter(x=df_cluster_unfolded['minwidth'],y=df_cluster_unfolded['tdh'],c=df_cluster_unfolded['kmeans_2'])
            ax14.title.set_text(' ')
            ax14.set_xlabel('minwidth')
            ax14.set_ylabel('tdh')

            ax20.scatter(x=df_cluster_unfolded['minwidth'],y=df_cluster_unfolded['leftpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax20.title.set_text(' ')
            ax20.set_xlabel('peakwidth')
            ax20.set_ylabel('leftpeak_n')

            ax21.scatter(x=df_cluster_unfolded['minwidth'],y=df_cluster_unfolded['rightpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax21.title.set_text(' ')
            ax21.set_xlabel('minwidth')
            ax21.set_ylabel('rightpeak')

            ax22.scatter(x=df_cluster_unfolded['tdh'],y=df_cluster_unfolded['leftpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax22.title.set_text(' ')
            ax22.set_xlabel('tdh')
            ax22.set_ylabel('leftpeak_n')

            ax23.scatter(x=df_cluster_unfolded['tdh'],y=df_cluster_unfolded['minwidth'],c=df_cluster_unfolded['kmeans_2'])
            ax23.title.set_text(' ')
            ax23.set_xlabel('tdh')
            ax23.set_ylabel('rightpeak')    

            ax24.scatter(x=df_cluster_unfolded['leftpeak'],y=df_cluster_unfolded['rightpeak'],c=df_cluster_unfolded['kmeans_2'])
            ax24.title.set_text(' ')
            ax24.set_xlabel('leftpeak')
            ax24.set_ylabel('rightpeak')    

            plt.show()
            plt.close()

            #plotting each recording of each fish and with corresponding k cluster as title to compare classification
            
            stand_amp_ = df_cluster_sorted['stand_amp']
            timepoints_ = df_cluster_sorted['timepoints']

            for a in range(len(stand_amp_)):
                'create subplot'
                wave_clust,((ax10,ax11,ax12,ax13,ax14),(ax20,ax21,ax22,ax23,ax24),(ax30,ax31,ax32,ax33,ax34),(ax40,ax41,ax42,ax43,ax44),(ax50,ax51,ax52,ax53,ax54),(ax60,ax61,ax62,ax63,ax64)) = plt.subplots(6,5)

                ax1_array = np.array([ax10,ax11,ax12,ax13,ax14])
                ax2_array = np.array([ax20,ax21,ax22,ax23,ax24])
                ax3_array = np.array([ax30,ax31,ax32,ax33,ax34])
                ax4_array = np.array([ax40,ax41,ax42,ax43,ax44])
                ax5_array = np.array([ax50,ax51,ax52,ax53,ax54])
                ax6_array = np.array([ax60,ax61,ax62,ax63,ax64])   
                
                idx_1 = []
                idx_2 = []
                idx_3 = []
                idx_4 = []
                idx_5 = []
                idx_6 = []
                'create index range for every row'
                for nb in range(len(stand_amp_[a])):
                    if nb < 5:
                        idx_1.append(nb)
                    if nb >=5 and nb <10:
                        idx_2.append(nb)
                    if nb >=10 and nb <15:
                        idx_3.append(nb)
                    if nb >=15 and nb <20:
                        idx_4.append(nb)
                    if nb >=20 and nb <25:
                        idx_5.append(nb)
                    if nb >=25:
                        idx_6.append(nb)

                'plotting'
                for i1 in range(len(idx_1)):
                    ix1 = idx_1[i1]
                    ax1_array[i1].title.set_text(str(df_cluster_sorted['kmeans_2'][a][ix1]))
                    ax1_array[i1].plot(timepoints_[a][ix1],stand_amp_[a][ix1])
                for i2 in range(len(idx_2)):
                    ix2 = idx_2[i2]
                    ax2_array[i2].plot(timepoints_[a][ix2],stand_amp_  [a][ix2])
                    ax2_array[i2].title.set_text(str(df_cluster_sorted['kmeans_2'][a][ix2]))
                for i3 in range(len(idx_3)):
                    ix3 = idx_3[i3]
                    ax3_array[i3].plot(timepoints_[a][ix3],stand_amp_[a][ix3])
                    ax3_array[i3].title.set_text(str(df_cluster_sorted['kmeans_2'][a][ix3]))
                for i4 in range(len(idx_4)):
                    ix4 = idx_4[i4]
                    ax4_array[i4].plot(timepoints_[a][ix4],stand_amp_[a][ix4])
                    ax4_array[i4].title.set_text(str(df_cluster_sorted['kmeans_2'][a][ix4]))
                for i5 in range(len(idx_5)):
                    ix5 = idx_5[i5]
                    ax5_array[i5].plot(timepoints_[a][ix5],stand_amp_[a][ix5])
                    ax5_array[i5].title.set_text(str(df_cluster_sorted['kmeans_2'][a][ix5]))
                for i6 in range(len(idx_6)):
                    ix6 = idx_6[i6]
                    ax6_array[i6].plot(timepoints_[a][ix6],stand_amp_[a][ix6])
                    ax6_array[i6].title.set_text(str(df_cluster_sorted['kmeans_2'][a][ix6]))

                'fig attributes'
                wave_clust.subplots_adjust(wspace=0.1, hspace=0.5)

                plt.show()
            
        elif data_type == 'waveform':
            print('lol')
           

def k_means_group_plot(timepoints:List,amplitude:List,kmeanlabels:List):
    """
    This function returns and subplot which includes one plot, 
    where all data with the same kmean labels are plottet over each other.

    Input: 
    timepoints: list of timepoints
    amplitude: list of amplitudes
    kmeanlabels: list of numbers that describe in which 
                kmean cluster each recording belongs too
    
    returns: Subplot of each kmean cluster including correspoding data
    """


    fig,ax = plt.subplots(1,3)

    for idx,km in enumerate(kmeans):
        if km == 0:
            ax[0].plot(timepoints[idx],amplitude[idx],alpha=0.5)
            ax[0].set_title('Cluster 1')
        if km == 1:
            ax[1].plot(timepoints[idx],amplitude[idx],alpha=0.5)
            ax[1].set_title('Cluster 2')
        else:
            ax[2].plot(timepoints[idx],amplitude[idx],alpha=0.5)
            ax[2].set_title('Cluster 3')
       
    fig.supxlabel = 'Time'
    fig.supylabel = 'Amplitude'
    fig.suptitle = 'Plot of EoDs in each Kmean cluster'


    # calculate hom many 


    #def kmeans_visualizer(amplitude):

def pca_accum_exvar(amplitude,freq):
    

    # PCA of data to see which amplitude at time x was most impectfull
    pca = PCA(n_components=10)  # We want to reduce to 2 components for plotting
    X_pca = pca.fit_transform(amplitude)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots(1,2)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    ax[0].plot(cumulative_variance_ratio, marker='o')
    ax[0].set_xlabel('Number of Principal Components')
    ax[0].set_ylabel('Cumulative Explained Variance Ratio')
    fig.suptitle('Cumulative Explained Variance Ratio by Principal Components')
    

    stand_freq = np.array(freq)/max(freq)
    fused_data = np.column_stack((amplitude,stand_freq))

    pcaf = PCA(n_components=10)
    X_pcaf = pcaf.fit_transform(fused_data)
    expl_rat = pcaf.explained_variance_ratio_
    cum_var = np.cumsum(expl_rat)
    ax[1].plot(cumulative_variance_ratio, marker='o')
    ax[1].set_xlabel('Number of Principal Components')
    ax[1].set_ylabel('Cumulative Explained Variance Ratio')
    ax[1].set_title('With Eod Frequency')
    plt.show()

def kmeans_including_freq(amplitude,frequency,k_kmeans,pca_n=10):
    '''
    This function performs k-clustering on waveform data with the inclusion of
    EOD frequrency as parameter. It returnd the k labels of each fish and the Xpca of PCA.

    Input: 
    amplitude: array of amplitude of EODf
    frequency: array or list of EODf for every fish
    k_means: The amount of clusters the kclustering divides the data into
    pca_n: Optional. Number of pca components that are used. Default is 10.

    Output:
    Retuns the label of each cluster for each fish but also the Xpca of the PCA

    '''

    stand_freq = (np.array(frequency)-min(frequency))/(max(frequency)-min(frequency))

    fused_data = np.column_stack((amplitude,stand_freq))

    pca = PCA(n_components=pca_n)
    X_pca = pca.fit_transform(fused_data)

    kmeans = KMeans(n_clusters=k_kmeans)
    kmeans.fit(X_pca)
    k_labels=kmeans.labels_
    
    return k_labels,X_pca 

def parameter_pca_influence(data):
    """
    Function calculates the most influancual 10 features of the pca_components 
    that explain the most varriance.

    Input: 
    data: list or array of data

    returns: 
    top_features: Top features of each PCA componend by indx
    feature value: eigenvactor (?) value of each feature of each pca component
    explained_var: number in % that show how much of the varriance is explained

    """
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(data)
    n_pcs = pca.components_.shape[0]

    parameter_id = list(range(len(data)))
    #print(f'Feateures: # {parameter_id[-1]}')
    

    explained_var = []
    top_features = []
    feature_value = []

    for id in range(len(pca.components_)):
        if pca.explained_variance_ratio_[id] > 0.01:
           
            explained_var.append(pca.explained_variance_ratio_[id])

            cop = pca.components_[id]

            comp_id_link = list(zip(cop,parameter_id))
          
            sorted_list = sorted(comp_id_link, key=lambda x:x[0])

            sorted_comp, sorted_id = zip(*sorted_list)

            top_features.append(sorted_id[:10])
            feature_value.append(sorted_comp[:10])

        else:
            break
    
    return top_features, feature_value, explained_var   

def kcluster_best_n(amplitude):


    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(amplitude)


    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X_pca)
        Sum_of_squared_distances.append(km.inertia_)    

    plt.figure()

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

##### Main functio4

if __name__ == "__main__":
    ### setting

    path_wavforms = '/home/kuehn/data/waveforms/thunderfish'
    df_data = thunderclusterer(path_wavforms,mode='best', data_type='waveform')

    timepoints = df_data["timepoints"].to_list()
    stand_amp = df_data['stand_amp'].to_list()
    kmeans = df_data["kmeans_3"].to_list()
    
   
    
    #k_means_group_plot(timepoints=timepoints,amplitude=stand_amp,kmeanlabels=kmeans)
    kcluster_best_n(stand_amp)

    freq = df_data['freq'].to_list()
    stand_freq = np.array(freq)/max(freq)

    pca_accum_exvar(stand_amp,freq)

    embed()
    
    """modes = sys.argv[1:] 
    print(modes)
    
    if 'best' in modes:
        if 'parameters' in modes:
            thunderclusterer(path_wavforms,mode='best', data_type='parameters')
        if 'waveform' in modes:
            thunderclusterer(path_wavforms,mode='best', data_type='waveform')
    else:"""

        
        
     #thunderclusterer(path_wavforms,mode='best', data_type='waveform')""