#!/usr/bin/env python
# encoding: utf-8
"""
MainRs.py

Code for analyzing resting state data (pupil + EEG)
Created by Stijn Nuiten on 2022-06-01.
Copyright (c) 2022 __MyCompanyName__. All rights reserved.
"""

import sys

sys.path.append("/usr/local/fsl/lib/python3.10/site-packages")
sys.path.append("smb://fmg-research.uva.nl/psychology$/Projects/Project%202023_vanGaal_2019-BC-11514_RestingState/RS%20-%20Copy/Samuel")

# Complicated ish solution to include all information theory measures into file but seemed easier 
# than trying to figure out the server configuration

import jpype
# Path to the JVM library (libjvm.so)
jvm_path = '/home/c12172812/Desktop/jdk-22.0.1/lib/server/libjvm.so'

# Path to the infodynamics.jar file
infodynamics_jar = '/home/c12172812/Desktop/environment/lib/python3.8/site-packages/pyspi/lib/jidt/infodynamics.jar'

# Start the JVM with both paths
jpype.startJVM(jvm_path, "-Djava.class.path=" + infodynamics_jar)

import os, datetime
from os import listdir

import multiprocessing

max_cpus = multiprocessing.cpu_count()
max_cpus = int(max_cpus / 5)
os.environ["OMP_NUM_THREADS"] = str(max_cpus)
os.environ["MKL_NUM_THREADS"] = str(max_cpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(max_cpus)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_cpus)
os.environ["OPENBLAS_NUM_THREADS"] = str(max_cpus)

import subprocess, logging
import datetime, time, math
import pickle

import re

import glob

import scipy as sp
import numpy as np

from subprocess import *
from pylab import *
from numpy import *
from math import *
from os import listdir

import multiprocessing as mp

import matplotlib

import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')

from IPython import embed as shell

import pandas as pd

import concurrent.futures
import itertools
import plotly.graph_objects as go

from tqdm import tqdm
from plotly.subplots import make_subplots

# from RestFirstLevel import RestSession
from Spectral_Analysis import Analysis_individual
from Spectral_Analysis import Analysis_group
from Connectivity_analysis import Connectivity_individual
from Connectivity_analysis import Machine_learning

person = 'Stijn'
network = '/home/c12172812/RS-Copy/'
baseDir = os.path.join(network, person)
        
firstLevel = True

subs_samuel = [\
    '01',
    '02',
    '03',
    '04',
    '05',
    '06',
    '07',
    '08',
    '09', 
    '10',
    '11',
    '12', 
    '13', 
    '14', 
    '15', 
    '16', 
    '17',
    '18', 
    '19',
    '20',
    '21', 
    '22', 
    '23', 
    '24', 
    '25', 
    '26', 
    '28', 
    '29', 
    '30'
    ]

# Use these for Stijn
subs_stijn = [\
    '06',
    '11',
    '12', 
    '13', 
    '14', 
    '15', 
    '16', 
    '17', 
    '18', 
    '19',
    '20',
    '22', 
    '23', 
    '24', 
    '25', 
    '26', 
    '27',
    '29', 
    '30'
    ]

# Needs to be adjusted for Samuel and Stijn when running the pre-processing
ids = ['RS2','RS3']

if person == "Stijn":
    IDs = ['{}_{}'.format(s,i) for s in subs_stijn for i in ids]

elif person == "Samuel":
    IDs = ['{}_{}'.format(s,i) for s in subs_samuel for i in ids]

ID_error = []

def run_pre(ID):
        rs = RestSession(baseDir=baseDir, ID=ID)
        # rs.preprocess_pupil()
        # rs.pup_data()
        # rs.eeg_data()
        # rs.combining_files()

def run_spectral_analysis_person(ID): 
     an = Analysis_individual(base_dir=baseDir, train_or_test= "Test", ID = ID)
     print("Train or Test Split")
     # an.train_test_split(0.8, 0.2)
     print("Spectral")
     an.spectral_analysis_per_subject()
     # an.correlation_arousal_channel(max_shift=1000, step_size= 10)
     an.correlation_arousal_channel(max_shift=5000)

def run_spectral_analysis_group(): 
    an = Analysis_group(base_dir=baseDir, train_or_test= "Train")
    # an.spectral_across_people()
    an.arousal_t_test()
    # an.histogram_duration()
    # an.arousal_eeg_correlation(max_shift = 5000)
    # an.plot_maximum_correlation_overall()

def run_connectivity_analysis_individual(ID): 
     an = Connectivity_individual(base_dir=baseDir, train_or_test= "Test", ID=ID)
     an.calculate_spi_per_measure()
     # an.calculate_spi_run_time()
     # an.create_yaml_files(base_yaml_filename = "/home/c12172812/Desktop/environment/lib/python3.8/site-packages/pyspi/personalised.yaml")
    
def run_machine_learning_connectivity(): 
    an = Machine_learning(base_dir=baseDir, train_or_test="Test")
    # an.get_data()
    # an.load_and_flatten_data()
    # an.run_machine_learning()
    # an.testing_connectivity_differences()
    # an.correlation_between_feature_and_difference()
    # an.normal_mvpa()
    # an.feature_investigation()
    # an.testing_on_new_data()
    an.correlate_differences_test_and_train()

if __name__ == '__main__':
    if firstLevel:
        for ID in tqdm(IDs):
            print(ID)
            # To run pre-processing for Stijn, please run the code in his code folder
            # as it was written specificially for Samuel due to differences in file naming 
            # events etc. 

            # run_pre(ID)
            # Not very efficient but want to keep the code running 
            # for both pre-processing and analysis 
            # and I do not want to change pre-processing anymore 
            # and using RS2 and RS3 is not necessary for the analysis
            # Also not needed as the server is hella fast 
            
            if "RS3" in ID: 
                 continue
            
            # run_spectral_analysis_person(ID)
        # run_connectivity_analysis_individual(ID)
        run_spectral_analysis_group()
        # shell()
        # run_machine_learning_connectivity()