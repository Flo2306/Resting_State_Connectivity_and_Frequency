#!/usr/bin/env python
# encoding: utf-8
"""
exp.py

Created by Stijn Nuiten on 2018-02-14.
Copyright (c) 2018 __MyCompanyName__. All rights reserved.
"""
import os, sys, datetime
from os import listdir
import subprocess, logging
import datetime, time, math
import pickle

import re

import glob

import scipy as sp
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import measurements
import numpy as np

from subprocess import *
from pylab import *
from numpy import *
from math import *
from os import listdir

from IPython import embed as shell

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.patches as patch

import mne
from mne.time_frequency import tfr_morlet
from mne import io, EvokedArray
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.viz import plot_evoked_topo

from mne.stats import spatio_temporal_cluster_test

# from functions.statsfuncs import cluster_ttest

import pyprep
# from autoreject import AutoReject
class EEG(object):
    def __init__(self, baseDir,ID=None,ext='',**kwargs):
        if kwargs.items():
            for argument in ['eegFilename','lims','bad_chans','event_ids']:
                value = kwargs.pop(argument, 0)
                setattr(self, argument, value)

        self.baseDir = baseDir
        self.chanSel = {}
        self.chanSel['ALL'] = None #self.epochs.info['ch_names'][0:64]
        self.chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
        self.chanSel['PAR'] = ['P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8']
        self.chanSel['FRO'] = ['Fp1', 'AF7', 'AF3', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz']
        self.chanSel['TMP'] = ['FT7', 'C5', 'T7', 'TP7', 'CP5', 'FT8', 'C6', 'T8', 'TP8', 'CP6']
        self.chanSel['OPA'] = ['P1', 'P3', 'P5', 'P7', 'P9', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3'
                               'O1', 'Iz', 'Oz', 'POz', 'PO8', 'PO4', 'O2', 'PO9', 'PO10' ]
        self.chanSel['CDA'] = ['P5', 'P6', 'P7', 'P8', 'PO7', 'PO8', 'O1', 'O2', 'PO9', 'PO10']
        self.ID = ID
        if self.ID != None:
            self.subject,self.index,self.task  = self.ID.split('_')
 
            self.plotDir =  os.path.join(self.baseDir,'figs','indiv',self.subject)
            if not os.path.isdir(self.plotDir):
                os.makedirs(self.plotDir) 
            try:
                extension = '.bdf'
                self.eegFilename = glob.glob(os.path.join(self.baseDir, 'Raw', self.task, self.subject,'*' + self.subject + '*' + str(self.index) + '*.bdf'))[-1]
                self.raw =  mne.io.read_raw_bdf(self.eegFilename, eog = ['HL','HR','VU','VD'],
                    misc = ['M1','M2'], preload=True)
            except:
                try:
                    extension = '.raw.fif'
                    self.eegFilename = glob.glob(os.path.join(self.baseDir, 'Raw', self.task, self.subject, '{}_{}_{}*.raw.fif'.format(self.subject, str(self.index),self.task)))[0]
                    self.raw =  mne.io.read_raw_fif(self.eegFilename,  preload=True)
                except:
                    print("RAW FILE NOT FOUND")

            try:
                self.epochFilename = glob.glob(os.path.join(self.baseDir, 'Proc', self.task, self.subject, ext,  '_'.join((self.subject,self.index,'epo.fif'))))[-1]                    
                self.epochs =  mne.read_epochs(self.epochFilename, preload=True)
                print( "epoch files found and loaded")
            except:
                print ("\n\n\n\nEpoch-file not found, run preprocessing first\n\n\n\n")


        else:
            self.plotDir =  os.path.join(self.baseDir,'figs','group')
            if not os.path.isdir(self.plotDir):
                os.makedirs(self.plotDir) 

    def preproc(self, outdir=None, ext='', baseline=None, epochTime=(-1.0, 2.0), ica_raw=True, reject=None, reject_by_annotation=False,
                overwrite = False, f_freq=None, detect_bad_chans = False, detect_bad_eps = False, plot_preproc_results=True):
        """ This method runs all the necessary pre-processing steps on the raw EEG-data. 
            Included are:
            - re-referencing
            - blink detection ()
            - creating epochs 
            - ICA (+ selection and removal)
        """
        preprocPlotDir = os.path.join(self.plotDir, ext, 'preproc')
        if plot_preproc_results:
            if not os.path.isdir(preprocPlotDir):
                os.makedirs(preprocPlotDir)


        if 'ECD' in self.raw.ch_names:
            self.raw.drop_channels(['ECD','ECU'])
        elif 'EXG7' in self.raw.ch_names:
            self.raw.drop_channels(['EXG7','EXG8'])

        self.raw.set_montage('biosemi64')

        try:   
            self.raw.set_eeg_reference(ref_channels = ['M1','M2'], projection=False)
        except:
            self.raw.set_eeg_reference(ref_channels = 'average', projection=False)

        if f_freq:
            self.raw.filter(f_freq[0], f_freq[1])
            
        if self.bad_chans:
            self.raw.info['bads'] = self.bad_chans 
        elif detect_bad_chans == 'RANSAC':
            nd = pyprep.NoisyChannels(self.raw.copy().pick_types(eeg=True),random_state=14)
            nd.find_bad_by_ransac(n_samples=25)
            self.raw.info['bads'].extend(nd.bad_by_ransac)
            print('Detected ' + str(self.raw.info['bads']) +' as bad channels (RANSAC)')
        elif detect_bad_chans == 'all':
            nd = pyprep.NoisyChannels(self.raw,random_state=14)
            nd.find_all_bads(ransac=False)
            self.raw.info['bads'].extend(nd.get_bads())
            print('Detected ' + str(self.raw.info['bads']) +' as bad channels (ALL)')
        elif detect_bad_chans == 'hf':
            nd = pyprep.NoisyChannels(self.raw,random_state=14)
            nd.find_bad_by_hf_noise(3.29053)
            self.raw.info['bads'].extend(nd.bad_by_hf_noise)
            print('Detected ' + str(self.raw.info['bads']) +' as bad channels (HF, 99.9%)')
        elif detect_bad_chans == 'dev':
            nd = pyprep.NoisyChannels(self.raw,random_state=14)
            nd.find_bad_by_deviation(3.29053)
            self.raw.info['bads'].extend(nd.find_bad_by_deviation)
            print('Detected ' + str(self.raw.info['bads']) +' as bad channels (deviation, 99.9%)')
        self.raw.info['int_bads'] = ''

        # For plotting demo Simon
        # self.raw.plot(n_channels=64,bad_color=(1, 0, 0), scalings=dict(eeg=40e-6),show=True)

        if len(self.raw.info['bads']) > 0:
            bads = self.raw.info['bads']
            self.raw.interpolate_bads(reset_bads=True) 
            self.raw.info['int_bads'] = bads
            
        if ica_raw:
            filt_raw = self.raw.copy()
            filt_raw.load_data().filter(l_freq=1., h_freq=None)
            ica = ICA(n_components=25, random_state=1)
            ica.fit(filt_raw)


            bad_idx, scores = ica.find_bads_eog(filt_raw, ch_name = 'VU')#, threshold=2)
            ica.exclude = bad_idx

            ica.plot_overlay(filt_raw, exclude=bad_idx, picks='eeg', show=False)
            plt.savefig(os.path.join(preprocPlotDir, 'ICA_cleaning_{}.pdf'.format(self.ID)))
            ica.apply(self.raw, exclude=bad_idx)


        if reject_by_annotation:        # Detect and remove blink artefacts
            eog_events = mne.preprocessing.find_eog_events(self.raw)
            n_blinks = len(eog_events)

            # Center to cover the whole blink with full duration of 0.5s:
            onset = eog_events[:, 0] / self.raw.info['sfreq'] - 0.25
            duration = np.repeat(0.5, n_blinks)
            self.raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                      orig_time=self.raw.info['meas_date'])

        picks_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=True,
                       stim=False)
        if not hasattr(self,'events'):
            self.events = mne.find_events(self.raw,shortest_event=1, consecutive=False)

        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_ids,
             preload=True, tmin = epochTime[0], tmax = epochTime[1], baseline = baseline, 
             picks=picks_eeg, reject_by_annotation=reject_by_annotation)
        if detect_bad_eps == 'auto':
            picks = mne.pick_types(self.epochs.info, meg=False, eeg=True,
                       stim=False, eog=False,
                       include=[], exclude=[])

            ar = AutoReject(picks=picks, random_state=42)
            ar.fit_transform(self.epochs.copy())
            rejected = ar.get_reject_log(self.epochs).bad_epochs
            self.epochs.events[rejected,2]=0
            self.epochs.event_id['rejected'] = 0            
            epochFilename = self.subject + '_' + str(self.index) + '_clean_epo.fif'
        else:
            epochFilename = self.subject + '_' + str(self.index) + '_epo.fif'

        if outdir == None:
            outdir = os.path.join(self.baseDir, 'Proc', self.task, self.subject)
        if not os.path.isdir(outdir):
            os.makedirs(outdir) 
        if not hasattr(self,epochFilename):
            self.epochFilename = os.path.join(outdir, epochFilename)
        self.epochs.save(self.epochFilename, overwrite = overwrite)




