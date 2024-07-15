#!/usr/bin/env python
# encoding: utf-8
"""
exp.py

Created by Stijn Nuiten on 2018-02-14.
Copyright (c) 2018 __MyCompanyName__. All rights reserved.
"""
import sys

sys.path.append("/usr/local/fsl/lib/python3.10/site-packages")

import os, datetime, shutil
import mne 
import pandas as pd 
import os 
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import subprocess, logging
import datetime, time, math
import pickle

import re
import glob

import scipy as sp
import scipy.stats as stats
import scipy.signal as signal
import numpy as np

from IPython import embed as shell

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
# shell()

# matplotlib.use('TkAgg')

#'GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import seaborn as sns

import itertools

import statsmodels
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf

import mne
# mne.viz.set_browser_backend('qt')
# from mne.time_frequency import tfr_morlet
# from mne.time_frequency import tfr_morlet, psd_multitaper
from mne import io, EvokedArray
from mne.datasets import sample
from mne.decoding import Vectorizer, get_coef
from mne.preprocessing import ICA
from mne.stats import spatio_temporal_cluster_test, permutation_cluster_1samp_test, permutation_cluster_test
from mne.filter import filter_data
from mne.channels import find_ch_adjacency

 
# from pyEEG.EEG import EEG
# from pyEEG import dva
# from pyEEG.MVPA import MVPA

# from cv import ShuffleBinLeaveOneOut
import autoreject
from autoreject import AutoReject

import preproc 
import pyprep

# from functions.statsfuncs import *
import hedfpy

# sns.set(style='ticks', font='Arial', font_scale=1, rc={
#     'axes.linewidth': 0.5,
#     'axes.labelsize': 12,
#     'axes.titlesize': 10,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 10,
#     'xtick.major.width': 1,
#     'ytick.major.width': 1,
#     'text.color': 'Black',
#     'axes.labelcolor':'Black',
#     'xtick.color':'Black',
#     'ytick.color':'Black',} )
# sns.set_palette('Blues_r')
# sns.plotting_context()



class RestSession(object):
    def __init__(self,baseDir, ID = None):
        self.ID = ID
        self.subject  = self.ID.split('_')[0]
        self.task = 'resting_state'
        self.index = str(self.ID.split('_')[-1])
        self.index_adjusted = ""

        if self.index == "RS2": 
            self.index_adjusted = "s2"
        elif self.index == "RS3": 
            self.index_adjusted = "s3"

        # shell(colors="neutral")

        self.baseDir = baseDir
        self.rawDir = os.path.join(self.baseDir, 'pupil')
        self.procDir = os.path.join(self.baseDir,'PROC', 'pupil', self.subject)
        self.procEEG = os.path.join(self.baseDir, 'PROC', 'eeg', self.subject)
        self.finalDir = os.path.join(self.baseDir, 'PROC', 'Combined', self.subject)

        if not os.path.isdir(self.procDir):
            os.makedirs(os.path.join(self.procDir))
        if not os.path.isdir(self.procEEG):
            os.makedirs(os.path.join(self.procEEG))
        if not os.path.isdir(self.finalDir):
            os.makedirs(os.path.join(self.finalDir))
        # self.drugs = pd.read_excel(os.path.join(self.baseDir,'drug_order.xlsx'))
       
        self.analysis_params = {
                        'sample_rate' : 500,
                        'lp' : 10.0,
                        'hp' : 0.01,
                        'normalization' : 'zscore',
                        'regress_blinks' : True,
                        'regress_sacs' : True,
                        'regress_xy' : False,
                        'use_standard_blinksac_kernels' : False,
                        }       
         
        self.sample_rate = self.analysis_params['sample_rate']

    def preprocess_pupil(self):
        # data:
        edf_file = os.path.join(self.rawDir, self.ID + ".edf")

        # folder hierarchy:
        # shell()
        # hdf5 filename:
        hdf5_filename = os.path.join(self.procDir, '{}.hdf5'.format(self.ID))
        try:
            os.remove(hdf5_filename)
        except OSError:
            pass
        
        # initialize hdf5 HDFEyeOperator:
        ho = hedfpy.HDFEyeOperator(hdf5_filename)
            
        alias = 'S{}_{}_0'.format(self.subject,self.index) #, run_nrs[i]
        
        # shell()
        shutil.copy(edf_file, os.path.join(self.procDir, '{}.edf'.format(alias)))
        # shell()
        ho.add_edf_file(os.path.join(self.procDir, '{}.edf'.format(alias)))

        ho.edf_message_data_to_hdf(alias=alias)
        # ho.close_hdf_file()
        ho.edf_gaze_data_to_hdf(alias=alias,
                                sample_rate=self.analysis_params['sample_rate'],
                                pupil_lp=self.analysis_params['lp'],
                                pupil_hp=self.analysis_params['hp'],
                                normalization=self.analysis_params['normalization'],
                                regress_blinks=self.analysis_params['regress_blinks'],
                                regress_sacs=self.analysis_params['regress_sacs'],
                                use_standard_blinksac_kernels=self.analysis_params['use_standard_blinksac_kernels'],
                                )
        ho.close_hdf_file()

    def load_h5(self):
        # Preset folders, filenames, aliases
        self.hdf5_filename = os.path.join(self.procDir, '{}.hdf5'.format(self.ID))
        self.ho = hedfpy.HDFEyeOperator(os.path.expanduser(self.hdf5_filename ))

        with pd.HDFStore(self.ho.input_object) as hf:
            datasetnames = hf.keys()
            self.aliases = np.unique([re.findall('(S{}_{}_\d)'.format(self.subject, self.index), st) for st in datasetnames])
        
        alias = 'S{}_{}_0'.format(self.subject, self.index) #, run_nrs[i]

        trial_phase_times = []
        trial_parameters = []
        session_start_EL_time = []
        session_stop_EL_time = []
        trial_times = []
        num_trials = 0 
        num_blocks = 0

        # Load trial times from hd5
        self.trial_times = self.ho.read_session_data(alias, 'block_0')

        # Also load all trial phase times
        self.trial_phase_times = self.ho.read_session_data(alias, 'blocks')

        # The first trial_phase_time is very short (81ms), so let's drop it
        self.trial_times = self.trial_times[1:]
        self.trial_phase_times = self.trial_phase_times[1:]

        self.session_start_EL_time = np.array(self.trial_times['time'])[0]
        self.session_stop_EL_time = np.array(self.trial_times['time'])[-1]

    def pup_data(self, signal = 'pupil_bp_clean'):
        
        # Replaced the used signal with pupil_bp_clean from the original pupil_bp_clean_zscore to make 
        # comparison between groups possible. Otherwise normalising within each participant and group makes 
        # baseline differences disappear 

        # signal = 'pupil_bp_clean_zscore'

        # Load in hdf5-file
        self.load_h5()

        sub_pupil_dir = os.path.join(self.procDir,self.task, self.subject)

        # relevant events
        event_dict = {'stim': 2}#, 'resp': 5}

        # Get info, for mne epoch creation
        info = mne.create_info(['x_gaze', 'y_gaze', 'pupil_raw', 'pupil_session_z', 'pupil_block_z'], sfreq=self.sample_rate, ch_types='misc', verbose=None)

        x_gaze = self.ho.signal_during_period(time_period=[0,np.inf], alias=self.aliases[0], signal='gaze_x_int')
        y_gaze = self.ho.signal_during_period(time_period=[0,np.inf], alias=self.aliases[0], signal='gaze_y_int')
        pupil = self.ho.signal_during_period(time_period=[0,np.inf], alias=self.aliases[0], signal=signal)
        block_t = self.ho.signal_during_period(time_period=[0,np.inf], alias=self.aliases[0], signal='time')
        
        # Now put all these time-series in one DataFrame
        eye_df = pd.concat(objs=[x_gaze, y_gaze, pupil, block_t],axis=1, names=['x_gaze', 'y_gaze', 'pupil_size', 'time'])

        # And trim off the time that is not of interest 
        eye_df = eye_df.loc[(eye_df['time'] >= self.session_start_EL_time) & (eye_df['time'] <= self.session_stop_EL_time)].reset_index(drop=True)
        eye_df.time -= eye_df['time'][0]
        self.eye_df = eye_df

        self.eye_df.to_csv(os.path.join(self.procDir,'{}_{}_eye_tracking.csv'.format(self.subject, self.index)))

    def eeg_data(self):

        # # First, check whether pupil data is loaded. If not, load
        # if not hasattr(self,'eye_df'):
        #     try:
        #         self.pup_data()
        #     except:
        #         raise OSError('Pupil data not found, run preprocess_pupil first')

        eegDir = os.path.join(self.baseDir, 'eeg')
       
        # eegFile = glob.glob(os.path.join(eegDir, '{}*{}'.format(self.ID, self.index_adjusted)))[0]
        # Define the pattern to search for
        eegFile = os.path.join(eegDir, '{}_resting_state_{}.bdf'.format(self.subject, self.index_adjusted))
        
        self.eeg_raw = mne.io.read_raw_bdf(eegFile, eog = ['HL','HR','VU','VD'],
                    misc = ['ML','MR','ECL','ECR',], preload=True)
        
        # self.eeg_raw.crop(tmax = 100)

        # shell()

        # shell(color='neutral')

        ### find possible start point of RESTING STATE and recode it 333
        #stp = np.where(self.eeg_events[:, 2] == 11)[0][0] - 11
        #if self.eeg_events[stp,2] == 125:
            #self.eeg_events[stp,2] = 333

        # shell(color='neutral')
        #find end point of resting state and recode it 444
        #eps = np.where(self.eeg_events[:, 2] == 143)[0]
        #for ep in eps:
            #try: 
                #if self.eeg_events[ep+1,2] == 128:
                    #self.eeg_events[ep+1,2] = 444
            #except:
                #if ep == len(self.eeg_events)-1:
                    #self.eeg_events[ep,2] = 444

        #ep = np.where(self.eeg_events[:,2] == 444)[0][0]
        
        # # find possible start point of BIKE STILL and recode it 555
        # stp = np.where(self.eeg_events[:, 2] == 11)[0] - 11
        # for stpp in stp:
        #     if self.eeg_events[stpp,2] == 126:
        #         self.eeg_events[stpp,2] = 555

        # # shell(color='neutral')
        # #find end point of bike and recode it 666
        # eps = np.where(self.eeg_events[:, 2] == 143)[0]
        # for ep in eps:
        #     try:
        #         if (self.eeg_events[ep+1,2] == 128) & (self.eeg_events[ep+1-599,2] == 555):
        #             self.eeg_events[ep+1,2] = 666
        #     except:
        #         if ep == len(self.eeg_events)-1:
        #             self.eeg_events[ep,2] = 666

        # stp = np.where(self.eeg_events[:,2]==555)[0][0]
        # ep = np.where(self.eeg_events[:,2] == 666)[0][0]
        
        #if ep-stp > 500:
            #st = self.eeg_events[stp,0]/self.eeg_raw.info['sfreq']
            #et = self.eeg_events[ep,0]/self.eeg_raw.info['sfreq']
            #self.eeg_raw.crop(tmin=st, tmax=et, include_tmax=True)

        # mapping = {125: 'start rest', 128: 'end block', 126: 'bike still', 127: 'bike move', 0:'0', 1: '1',
        #            2: '2', 3: '3', 4: '4', 5: '5',143:'143',555:'bike still'}

        # annot_from_events = mne.annotations_from_events(
        #     events=self.eeg_events, event_desc=mapping, sfreq=self.eeg_raw.info['sfreq'],
        #     orig_time=self.eeg_raw.info['meas_date'])
        # self.eeg_raw.set_annotations(annot_from_events)
        #
        # self.eeg_raw.plot(block=True)

        # Drop reset triggers
        # self.eeg_events = self.eeg_events[self.eeg_events[:,2]!=3840]

        # Drop run start (3965) and run end triggers (3968)
        # self.eeg_events = self.eeg_events[~ np.isin(self.eeg_events[:,2], [3965, 3968])]


        # TEST DURATION OF PUPIL AND EEG TRACES:
        #eeg_duration = (self.eeg_events[-1,0] - self.eeg_events[0,0])/self.eeg_raw.info['sfreq']
        # pup_duration =  (self.session_stop_EL_time -  self.session_start_EL_time)/1000

        # print('EEG trace duration: {}s'.format(eeg_duration))
        # print('Eye trace duration: {}s'.format(pup_duration))

        # if 'ECD' in self.eeg_raw.ch_names:
        #     self.eeg_raw.drop_channels(['ECL','ECU'])
        # elif 'EXG7' in self.eeg_raw.ch_names:
        #     self.eeg_raw.drop_channels(['EXG7','EXG8'])

        # shell()

        # Define the channel types for all channels
        channel_types = {
            'EXG1': 'eog',   # EXG channel designated as EOG
            'EXG2': 'eog',   # EXG channel designated as EOG
            'EXG3': 'eog',   # EXG channel designated as EOG
            'EXG4': 'eog',   # EXG channel designated as EOG
            'EXG5': 'eog',   # EXG channel designated as EOG
            'EXG6': 'eog',   # EXG channel designated as EOG
            'EXG7': 'eog',   # EXG channel designated as EOG
            'EXG8': 'eog'    # EXG channel designated as EOG
            # Add channel types for all channels here
        }

        # Set the channel types for each channel
        self.eeg_raw.set_channel_types(channel_types)

        self.eeg_raw.set_montage("biosemi64")

        self.eeg_raw.filter(.01, None)
        self.eeg_raw.notch_filter(freqs=50)

        nd = pyprep.NoisyChannels(self.eeg_raw.copy().pick_types(eeg=True),random_state=14)
        nd.find_bad_by_ransac(n_samples=25)
        self.eeg_raw.info['bads'].extend(nd.bad_by_ransac)
        # print('Detected ' + str(self.eeg_raw.info['bads']) +' as bad channels (RANSAC)')

        self.eeg_raw.interpolate_bads(reset_bads=True, mode='accurate')

        # self.filt_raw.plot(show_scrollbars=False)
        self.filt_raw = self.eeg_raw.copy().filter(l_freq=1., h_freq=None)
        ica = ICA(n_components=25, random_state=1)
        ica.fit(self.filt_raw)
        # ica

        # shell(color="neutral")

        # self.eeg_raw.load_data()
        # ica.plot_sources(self.filt_raw, show_scrollbars=True, block=True)

        bad_idx, scores = ica.find_bads_eog(self.filt_raw, ch_name = 'EXG1')#, threshold=2)
        ica.exclude = bad_idx

        ica.plot_overlay(self.filt_raw, exclude=bad_idx, picks='eeg', show=False)
        plt.savefig(os.path.join(self.procDir, 'ICA_cleaning_{}.pdf'.format(self.ID)))
        ica.apply(self.eeg_raw, exclude=bad_idx)

        eog_events = mne.preprocessing.find_eog_events(self.eeg_raw)
        n_blinks = len(eog_events)

        self.eeg_raw.set_eeg_reference()

        # Center to cover the whole blink with full duration of 0.5s:
        onset = eog_events[:, 0] / self.eeg_raw.info['sfreq'] - 0.25
        duration = np.repeat(0.5, n_blinks)
        # self.eeg_raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
        #                           orig_time=self.eeg_raw.info['meas_date'])

        df = self.eeg_raw.to_data_frame(picks=['eeg'])

        df.to_csv(os.path.join(self.procEEG, self.procEEG,'{}_{}_resting.txt'.format(self.subject, self.index)),sep='\1',encoding='ascii')

        # I am using .fif as using edf after MNE means removing all non-EEG channels such as EOG channels. 
        self.eeg_raw.save(os.path.join(self.procEEG, '{}_{}_resting_raw.fif'.format(self.subject, self.index)), overwrite=True)

        # ica.plot_components()

    def combining_files(self): 

        # At this point I am very uncertain if this does what we want it to? Very weird combination of code
        # but partially due to kinda shitty data collection. Major part will also be just me messing up the code

        # Find when event 1 takes place in the eye-tracking. I could not figure out 
        # a solution using hedfpy so I am using this instead as it does not add 
        # a lot of complexity (still some but I think the server will manage)

        sampling_frequency_eye = 500

        eye_path_start = os.path.join(self.procDir, "S{}_{}_0.asc".format(self.subject, self.index))

        eye_tracking_start = mne.io.read_raw_eyelink(eye_path_start)

        # Extract annotations from eye-tracking data
        annotations_start = eye_tracking_start.annotations

        # Extract occurrence times of event "1"
        event_1_times = int(annotations_start.onset[annotations_start.description == "1"][0] * sampling_frequency_eye)

        # Load eye-tracking data
        eye_path = os.path.join(self.procDir, "{}_{}_eye_tracking.csv".format(self.subject, self.index))

        df = pd.read_csv(eye_path)
        
        eye_tracking_data = df['L_pupil_bp_clean'].values 

        # Due to differences in indexing to make sure both have a length of 5 minutes + initial value (index 0)
        eye_tracking_five_mins = event_1_times + 5 * 60 * sampling_frequency_eye + 1

        eye_tracking_data = eye_tracking_data[event_1_times:eye_tracking_five_mins] 

        normalised_eye_tracking_data= (eye_tracking_data - np.min(eye_tracking_data)) / (np.max(eye_tracking_data) - np.min(eye_tracking_data))  # Normalize to [0, 1]

        # Load EEG data
        eeg_path = os.path.join(self.procEEG, '{}_{}_resting_raw.fif'.format(self.subject, self.index))

        #eeg_path = '/Volumes/psychology$/Projects/Project 2023_vanGaal_2019-BC-11514_RestingState/RS - Copy/Stijn/PROC/eeg/11/11_1_resting_raw.fif'
        eeg_raw = mne.io.read_raw_fif(eeg_path, preload=True)

        # Resample EEG data to 500 Hz
        eeg_raw.resample(500, npad="auto", window="boxcar")

        #eeg_raw.plot()
        #plt.show()

        try:
            # Find events in the EEG data
            eeg_events = mne.find_events(eeg_raw, shortest_event=1, consecutive=True)
            if len(eeg_events) == 1: 
                return
        except: 
            print("No events in EEG file, skipping processing for this file")
            return
        
        try: 
            # Find indices of event 3840
            event_3840_indices = (eeg_events[:, 2] == 1).nonzero()[0]

            print(event_3840_indices)

            print(eeg_events)

            # Time of the first occurrence of event 3840
            eeg_start = eeg_events[event_3840_indices[0], 0] / eeg_raw.info['sfreq']

            eeg_end = eeg_start + 5 * 60 

            # Time of the last occurrence of event 3840
            # eeg_end = eeg_events[event_3840_indices[-2], 0] / eeg_raw.info['sfreq']

        except: 
            event_3840_indices = (eeg_events[:, 2] == 3840).nonzero()[0]

            print(event_3840_indices)

            print("FUCK???")

            # Time of the first occurrence of event 3840
            eeg_start = eeg_events[event_3840_indices[1], 0] / eeg_raw.info['sfreq']

            # Time of the last occurrence of event 3840
            #eeg_end = 5 * 60 * eeg_raw.info['sfreq']
            eeg_end = eeg_start + 5 * 60 

            print(eeg_start)

        print("Length of EEG Data per crop: ", eeg_raw.n_times / eeg_raw.info['sfreq'])

        # Crop the EEG data
        eeg_raw_crop = eeg_raw.copy().crop(tmin=eeg_start, tmax=eeg_end)

        # Trim EEG data to match the duration of eye-tracking data
        min_duration_samples = min(len(normalised_eye_tracking_data), eeg_raw_crop.n_times)

        print(min_duration_samples)
        
        eye_tracking_data_trimmed = normalised_eye_tracking_data[:min_duration_samples]
        
        print("Length of eye-tracking: ", len(normalised_eye_tracking_data))
        print("Length of EEG Data: ", eeg_raw_crop.n_times)

        # Create an info object for the eye-tracking data
        # Currently entering it as a ecog channel to make scaling of value easier for plotting and 
        # keep it separate from eog
        
        info_eye = mne.create_info(ch_names=['Pupil'], sfreq=sampling_frequency_eye, ch_types=['ecog'])

        # Create a RawArray object for the eye-tracking data
        raw_eye = mne.io.RawArray(eye_tracking_data_trimmed.reshape(1, -1), info_eye)

        # Add eye-tracking data as an EOG channel to the EEG data
        eeg_raw_crop.add_channels([raw_eye], force_update_info=True)

        eeg_raw_crop.save(os.path.join(self.finalDir, '{}_{}_resting_raw.fif'.format(self.subject, self.index)), overwrite=True)




        