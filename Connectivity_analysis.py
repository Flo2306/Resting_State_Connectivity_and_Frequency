import sys

# Set up for the path to find all modules
sys.path.insert(0, "/home/c12172812/Desktop/jdk-22.0.1/bin")
sys.path.insert(0, "/home/c12172812/Desktop/environment/lib/python3.8/site-packages")

import numpy as np
import mne
import os
import glob
import matplotlib.pyplot as plt 
import time
import pandas as pd
import psutil
import yaml
import random
import re
import time
import seaborn as sns

from tqdm import tqdm
from pyspi.calculator import Calculator

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.decomposition import PCA

from scipy.stats import ttest_rel, pearsonr
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import linkage, dendrogram
from itertools import cycle

class Connectivity_individual(object):
    """
    A class used to calculate and handle connectivity measures for an individual.

    Attributes:
    -----------
    base_dir : str
        The base directory where results are stored.
    train_or_test : str
        Indicates whether the data is for training or testing.
    ID : str, optional
        The ID of the individual.
    """

    def __init__(self, base_dir, train_or_test, ID=None):
        """
        Initialize the Connectivity_individual class with the given parameters.

        Parameters:
        -----------
        base_dir : str
            The base directory where results are stored.
        train_or_test : str
            Indicates whether the data is for training or testing.
        ID : str, optional
            The ID of the individual.
        """
        self.ID_full = ID
        self.ID = self.ID_full.split("_")[0]
        self.base_dir = base_dir
        self.train_or_test = train_or_test
        self.sam_or_stijn = os.path.basename(self.base_dir)
        
        # Set the directory for storing connectivity results based on train_or_test
        if self.train_or_test == "Train":
            self.connectivity_results = os.path.join(self.base_dir, "Results", "Train", "Connectivity", self.ID)
        elif self.train_or_test == "Test":
            self.connectivity_results = os.path.join(self.base_dir, "Results", "Test", "Connectivity", self.ID)
        
        # Create the connectivity results directory if it does not exist
        if not os.path.exists(self.connectivity_results):
            os.makedirs(self.connectivity_results)

    def create_yaml_files(self, base_yaml_filename, output_directory="a", search_pattern="statistics*.yaml"):
        """
        Create YAML files for each sub-measure from a base YAML file.

        Parameters:
        -----------
        base_yaml_filename : str
            The filename of the base YAML file containing all measures.
        output_directory : str, optional
            The directory where the output YAML files will be saved. Default is "a".
        search_pattern : str, optional
            The pattern to search for YAML files in the output directory. Default is "statistics*.yaml".

        Returns:
        --------
        list
            A list of paths to the created YAML files matching the search pattern.
        """
        # Use the base YAML file's directory if output_directory is "a"
        if output_directory == "a":
            output_directory = os.path.dirname(base_yaml_filename)

        # Load the base YAML content from a file
        with open(base_yaml_filename, 'r') as file:
            data = yaml.safe_load(file)

        # Change directory to the output directory
        os.chdir(output_directory)

        # Iterate over the keys and save each sub-measure to a separate file
        for key, value in data.items():
            sub_measure_name = key.strip(".")
            filename = f"{sub_measure_name}.yaml"
            with open(filename, 'w') as file:
                yaml.dump({key: value}, file, default_flow_style=False)

        print("YAML files created successfully.")

        # Use glob to find all YAML files matching the search pattern
        yaml_files = glob.glob(f"{output_directory}/{search_pattern}")

        return yaml_files

    def calculate_spi_per_measure(self, max_run_time = 60): 
        """
        Calculate SPI (Spectral Power Interaction) per measure for EEG data.
        
        This function processes EEG files based on whether the data is for training or testing,
        and computes connectivity values for each measure defined in the YAML configuration files.

        The results are saved in the respective directories under connectivity results.

        Parameters: 

        max_run_time : int
            The maximum time it should take to compute one file in minutes. We currently set it
            at 60 minutes. 
        """

        if self.train_or_test == "Train":
            # Retrieve all EEG files for training data
            eeg_files = glob.glob(os.path.join(self.base_dir, "PROC", "Train_Data", "**", '**', '*.fif'), recursive=True)
        else: 
            # Retrieve all EEG files for testing data
            eeg_files = glob.glob(os.path.join(self.base_dir, "PROC", "Test_Data", "**", '**', '*.fif'), recursive=True)

        # Retrieve all YAML configuration files
        yaml_files = glob.glob("/home/c12172812/Desktop/environment/lib/python3.8/site-packages/pyspi/Yaml_files_per_measure/statistics*.yaml")

        # Separate files into two lists: one for 'infotheory' and one for others
        infotheory_files = [file for file in yaml_files if "infotheory" in file]
        other_files = [file for file in yaml_files if "infotheory" not in file]

        # Combine lists: other files first, then infotheory files
        yaml_files = other_files + infotheory_files

        # Ensure that "basic" files are run first
        basic_files = [file for file in yaml_files if "basic" in file]
        yaml_files = [file for file in yaml_files if "basic" not in file]
        yaml_files = basic_files + yaml_files

        # Find a "base" EEG file to ensure that all data is selected in the correct order
        for file_path in eeg_files:
            raw = mne.io.read_raw_fif(file_path, preload=True)

            # Filter to select only EEG channels directly
            picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, ecog=False)

            # Get the channel names and the number of EEG channels
            eeg_channel_names_base = [raw.ch_names[i] for i in picks_eeg]
            num_eeg_channels = len(eeg_channel_names_base)

            if num_eeg_channels == 64:
                break

        # For testing if it works
        # yaml_files = yaml_files[1:3]
        # eeg_files = eeg_files[1:3]

        for yaml_file in tqdm(yaml_files):
            for i, time_frame in enumerate(range(8)):
                measure_name = os.path.basename(yaml_file).replace('.yaml', '')
                print(measure_name)
                
                if "config" not in measure_name: 
                    # Skipping all the wrong ones
                    continue   

                # Skipping all non-infotheory ones for the first loop to get some data and finish the process
                if "infotheory" not in measure_name: 
                    continue
            
                for j, file_path in enumerate(eeg_files):
                    if i == 0 and j == 0:
                        start_time = time.time()

                    filename = os.path.basename(file_path)
                    subject = filename.split('_')[0]
                    condition = filename.split('_')[1]

                    raw = mne.io.read_raw_fif(file_path, preload=True)
                    raw.interpolate_bads(reset_bads=True, mode='accurate')

                    # Filter to select only EEG channels directly
                    raw.pick_channels(eeg_channel_names_base)

                    # Define the new sampling frequency (e.g., 128 Hz)
                    new_sampling_freq = 128

                    # Downsample the data
                    raw_resampled = raw.resample(sfreq=new_sampling_freq)

                    # Extract data as a NumPy array
                    eeg_data = raw_resampled.get_data()

                    max_time = 30 * (time_frame + 1)
                    min_time = 30 * time_frame
                    max_time_samples = max_time * new_sampling_freq
                    min_time_samples = min_time * new_sampling_freq

                    # Ensure the code continues running even if one file does not have enough data
                    if max_time_samples > eeg_data.shape[1]: 
                        continue

                    # Shorten the data for faster processing during coding
                    eeg_data = eeg_data[:, min_time_samples : max_time_samples]

                    # Initializing calculator
                    calc = Calculator(dataset=eeg_data, configfile=yaml_file)
                    
                    # Computing connectivity values, will take a long time 
                    calc.compute()

                    for key in calc.spis.keys(): 
                        measure_results = os.path.join(self.connectivity_results, key)
                        if not os.path.exists(measure_results):
                            os.makedirs(measure_results)

                        key_df = calc.table[key]
                        key_df.to_csv(os.path.join(measure_results, f"{key}_{subject}_{condition}_{min_time}_to_{max_time}.csv"))

                    if i == 0 and j == 0:
                        end_time = time.time()
                        elapsed_time = (end_time - start_time) / 60  # Calculate elapsed time in minutes
                    else: 
                        elapsed_time = 0

                    if elapsed_time > max_run_time:  # Check if iteration took longer than 60 minutes
                        print("Breaking the loop due to long processing time.")
                        break

                if elapsed_time > max_run_time:  # Additional check to break the outer loop if needed
                    print(f"The following method took too long to run (more than 60 mins for one file): {measure_name}")
                    break

    def calculate_spi_per_person(self): 
        """
        THIS CODE HAS NOT BEEN UPDATED AND NEEDS TO BE REVIEWED

        Calculate SPI per person for EEG data.
        
        This function processes EEG files based on whether the data is for training or testing,
        and computes connectivity values for each individual using a specified YAML configuration file.

        The results are saved in the respective directories under connectivity results.
        """

        raise ValueError("THIS FUNCTION HAS NOT BEEN UPDATED AND NEEDS TO BE REVIEWED/UPDATED")

        if self.train_or_test == "Train":
            # Retrieve all EEG files for the specified individual in training data
            eeg_files = glob.glob(os.path.join(self.base_dir, "PROC", "Train_Data", self.ID, '**', '*.fif'), recursive=True)
        else: 
            # Retrieve all EEG files for the specified individual in testing data
            eeg_files = glob.glob(os.path.join(self.test_folder, '**', '*.fif'), recursive=True)
        
        for i, file_path in enumerate(eeg_files):
            filename = os.path.basename(file_path)
        
            subject = filename.split('_')[0]
            condition = filename.split('_')[1]

            raw = mne.io.read_raw_fif(file_path, preload=True)

            # Filter to select only EEG channels directly
            picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, ecog=False)
            raw.pick_channels([raw.ch_names[i] for i in picks_eeg])

            # Define the new sampling frequency (e.g., 128 Hz)
            new_sampling_freq = 128

            # Downsample the data
            raw_resampled = raw.resample(sfreq=new_sampling_freq)

            # Extract data as a NumPy array
            eeg_data = raw_resampled.get_data()

            max_time = 30 * new_sampling_freq

            # Shorten the data for faster processing during coding 
            eeg_data = eeg_data[:, :max_time]

            # Initializing calculator
            calc = Calculator(dataset=eeg_data, configfile="/home/c12172812/Desktop/environment/lib/python3.8/site-packages/pyspi/personalised.yaml")
            
            # Computing connectivity values, will take a long time 
            calc.compute()

            for key in calc.spis.keys(): 
                measure_results = os.path.join(self.base_dir, "Results", "Connectivity", self.ID, key)
                if not os.path.exists(measure_results):
                    os.makedirs(measure_results)
            
                key_df = calc.table[key]
                print(key_df.shape)
                time.wait(5)
                key_df.to_csv(os.path.join(measure_results, f"{key}_{subject}_{condition}.csv"))

    def calculate_spi_run_time(self):
        """
        Calculate the runtime of SPI calculation for different data point increments.

        This function measures the processing time for different increments of data points in an EEG file
        to understand how the runtime scales with the amount of data.

        The results are plotted and saved as a PDF file.
        """

        # Define data points increments
        data_point_increments = [200, 250, 300, 350, 400]
        times = []

        # Retrieve the first EEG file for the specified individual in training data
        eeg_file = glob.glob(os.path.join(self.base_dir, "PROC", "Train_Data", self.ID, '**', '*.fif'), recursive=True)[0]

        # Load the EEG file
        raw = mne.io.read_raw_fif(eeg_file, preload=True)
        raw.set_montage("biosemi32")
        picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, ecog=False)
        raw.pick_channels([raw.ch_names[i] for i in picks_eeg])

        # Resample the data to 128 Hz
        raw_resampled = raw.resample(sfreq=128)
        eeg_data_full = raw_resampled.get_data()

        # Loop through each increment of data points
        for points in data_point_increments:
            start_time = time.time()
            eeg_data = eeg_data_full[:, :points]

            # Initialize the calculator with the EEG data
            calc = Calculator(dataset=eeg_data, configfile="/home/c12172812/Desktop/environment/lib/python3.8/site-packages/pyspi/personalised.yaml")
            
            # Compute connectivity values
            calc.compute()

            end_time = time.time()
            process_time = end_time - start_time
            times.append(process_time)

            print(f"Processed {points} data points in {process_time:.4f} seconds")

        # Plotting the results
        plt.figure(figsize=(8, 5))
        plt.plot(data_point_increments, times, marker='o', linestyle='-', color='b')
        plt.title('Processing Time vs. Data Points')
        plt.xlabel('Number of Data Points')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        # Save the plot as a PDF file
        plt.savefig(os.path.join(self.base_dir, "Results", "Connectivity", "run_time.pdf"))

class Machine_learning(object):
    """
    A class to set up directories and load data for machine learning tasks related to EEG connectivity measures.
    
    Attributes:
    -----------
    base_dir : str
        The base directory where results and data are stored.
    train_or_test : str
        Indicates whether the data is for training or testing.
    connectivity_folder : str
        Path to the folder containing connectivity results.
    connectivity_overall : str
        Path to the folder for overall connectivity results.
    connectivity_graphs : str
        Path to the folder for connectivity graphs.
    connectivity_models : str
        Path to the folder for connectivity models.
    connectivity_feature_importance : str
        Path to the folder for feature importance results.
    connectivity_significance_testing : str
        Path to the folder for significance testing results.
    connectivity_significant_differences : str
        Path to the folder for heat maps of significant differences.
    connectivity_feature_correlations : str
        Path to the folder for feature correlation graphs.
    channel_names : list
        List of channel names extracted from the EEG data.
    """

    def __init__(self, base_dir, train_or_test):
        """
        Initialize the Machine_learning class with the given parameters and set up necessary directories.

        Parameters:
        -----------
        base_dir : str
            The base directory where results and data are stored.
        train_or_test : str
            Indicates whether the data is for training or testing.
        """
        self.base_dir = base_dir
        self.train_or_test = train_or_test
        
        if self.train_or_test == "Train":
            self.connectivity_folder = os.path.join(self.base_dir, "Results", "Train", "Connectivity")
        elif self.train_or_test == "Test":
            self.connectivity_folder = os.path.join(self.base_dir, "Results", "Test", "Connectivity")
        else:
            raise ValueError("Please enter 'Test' or 'Train'")

        self.connectivity_overall = os.path.join(self.connectivity_folder, "Overall")
        self.connectivity_graphs = os.path.join(self.connectivity_folder, "Graphs")
        self.connectivity_models = os.path.join(self.connectivity_folder, "Models")
        self.connectivity_feature_importance = os.path.join(self.connectivity_models, "Feature_Importance")
        self.connectivity_significance_testing = os.path.join(self.connectivity_models, "Significance_Testing")
        self.connectivity_significant_differences = os.path.join(self.connectivity_graphs, "Heat_Maps_for_Differences")
        self.connectivity_feature_correlations = os.path.join(self.connectivity_graphs, "Feature_Correlations")

        # Retrieve the first EEG file for the specified individual in training data
        file_path = glob.glob(os.path.join(self.base_dir, "PROC", "Train_Data", "**", '**', '*.fif'), recursive=True)[0]

        # Load the raw data
        raw = mne.io.read_raw_fif(file_path, preload=True)

        # Pick only EEG channels
        raw.pick_types(eeg=True)

        # Extract channel names for use in multiple functions
        self.channel_names = raw.info['ch_names']

        # Create necessary directories if they do not exist
        if not os.path.exists(self.connectivity_overall):
            os.makedirs(self.connectivity_overall)

        if not os.path.exists(self.connectivity_graphs):
            os.makedirs(self.connectivity_graphs)
        
        if not os.path.exists(self.connectivity_models):
            os.makedirs(self.connectivity_models)

        if not os.path.exists(self.connectivity_feature_importance):
            os.makedirs(self.connectivity_feature_importance)
        
        if not os.path.exists(self.connectivity_significance_testing):
            os.makedirs(self.connectivity_significance_testing)

        if not os.path.exists(self.connectivity_significant_differences):
            os.makedirs(self.connectivity_significant_differences)

        if not os.path.exists(self.connectivity_feature_correlations):
            os.makedirs(self.connectivity_feature_correlations)

    def get_data(self):
        """
        Retrieve a list of valid methods that have exactly three files for each participant.
        
        This function scans through the connectivity folders of participants and checks for methods
        that have the required number of files, excluding certain folders.

        Returns:
        --------
        list
            A list of valid methods.
        """
        # Get list of participant folders
        participants = [os.path.join(self.connectivity_folder, participant) for participant in os.listdir(self.connectivity_folder) if os.path.isdir(os.path.join(self.connectivity_folder, participant))]
        
        # Dictionary to store methods with a count of participants having exactly three files
        methods_count = {}

        for participant in participants:
            # Exclude additional folders created in connectivity
            if participant.endswith("Overall"):
                continue
            if participant.endswith("Graphs"):
                continue
            if participant.endswith("Models"): 
                continue

            # Get list of method folders for each participant
            method_folders = [os.path.join(participant, method) for method in os.listdir(participant) if os.path.isdir(os.path.join(participant, method))]
            
            for method in method_folders:
                # Get list of files in each method folder
                files = [file for file in os.listdir(method) if os.path.isfile(os.path.join(method, file))]
            
                if len(files) > 4:
                    method_name = os.path.basename(method)
                    if method_name in methods_count:
                        methods_count[method_name] += 1
                    else:
                        methods_count[method_name] = 1

        # Find methods that have three files for every participant
        # -3 because length is always 1 longer + overall, graphs, and models not being relevant but still being in the length of participants
        valid_methods = [method for method, count in methods_count.items() if count == len(participants) - 3]

        print(valid_methods)

        return valid_methods

    def load_and_flatten_data(self):
        """
        Load and flatten the data for each valid method.

        This function retrieves the valid methods, loads the data for each participant,
        flattens the data, and saves it to the overall connectivity directory as CSV files.
        """
        methods = self.get_data()

        for method in tqdm(methods):
            method_data = []
            participants = [os.path.join(self.connectivity_folder, participant) for participant in os.listdir(self.connectivity_folder) if os.path.isdir(os.path.join(self.connectivity_folder, participant))]
            
            for participant in participants:  
                if participant.endswith("Overall"):
                    continue 
                if participant.endswith("06"):
                    continue
                if participant.endswith("Graphs"):
                    continue
                if participant.endswith("Models"): 
                    continue

                method_folder = os.path.join(participant, method)
                files = [os.path.join(method_folder, file) for file in os.listdir(method_folder) if os.path.isfile(os.path.join(method_folder, file))]
                
                for file in files:
                    if "to" not in file: 
                        continue 
                    if "ATX" in file:
                        condition = "ATX"
                    elif "DNP" in file:
                        condition = "DNP"
                    elif "PCB" in file:
                        condition = "PCB"
                    else: 
                        continue

                    file_name = file.split("/")[-1]
                    numbers = re.findall(r'\d+', file_name)
                    numbers_str = '_'.join(numbers)

                    df = pd.read_csv(file, header=None)  # Assuming no header and using pandas to load the data
                    df = df.iloc[1:, 1:]  # Exclude the first row and first column
                    np.fill_diagonal(df.values, 1)  # Filling the diagonal with 1 consistently for all methods
                    flattened_data = df.values.flatten()  # Flatten the DataFrame to a 1D array
                    flattened_df = pd.DataFrame([flattened_data])  # Convert the 1D array to a DataFrame
                    flattened_df['Condition'] = condition  # Add the condition column
                    flattened_df['Participant_time'] = numbers_str
                    method_data.append(flattened_df)
            
            combined_method_data = pd.concat(method_data, ignore_index=True)
            combined_method_data.to_csv(os.path.join(self.connectivity_overall, f"{method}.csv"))

def run_machine_learning(self): 
        """
        Run machine learning models on EEG connectivity data, evaluate their performance, 
        and analyze feature importance.
        """
        print("Getting Data")

        # Assuming self.connectivity_overall is a directory path
        csv_files = glob.glob(os.path.join(self.connectivity_overall, "*.csv"))

        print("Running Machine Learning")

        # List to store method names and their accuracies and AUCs
        results = []

        # Number of folds for cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Models to evaluate
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=5000, random_state=42),
            'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
            'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42)
        }

        for method in csv_files: 
            data_ml = pd.read_csv(method)

            # Replace remaining NaNs with 0
            data_ml = data_ml.fillna(0)

            if data_ml.shape[0] < 100: 
                continue

            if "prec_OAS" not in method:
                continue

            if 'Unnamed: 0' in data_ml.columns:
                data_ml = data_ml.drop(columns=['Unnamed: 0'])

            y = data_ml['Condition']
            x = data_ml.drop(columns=['Condition', 'Participant_time'])

            # Encode target labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Some code in here is to make sure the model is not just doing weird things due to its high 
            # performance. Randomly shuffling shows it is not due to the algorithm picking up noise, there 
            # are systematic differences. Could be due to participants moving differently in the conditions? 
            # Maybe increased arousal translates to more movement which leads to more issues in the EEG? 
            
            # np.random.shuffle(y_encoded)

            # Model performs well on a crazy train-test split like 10/90 and continues to improve as it is increased towards a 
            # more normal split like 80/20. From what it seems, the model just picks up a lot of signal, source of said signal 
            # needs to be investigated but I am kind of doing that by looking at the correlations between the two channels. 

            X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

            for model_name, model in models.items():
                print(f"Evaluating {model_name} for {method}")

                # if model_name == "LogisticRegression": 
                #     scaler = StandardScaler()
                #     X_train = scaler.fit_transform(X_train)
                #     X_test = scaler.transform(X_test)
                #     x = scaler.transform(x)

                try:
                    model.fit(X_train, y_train)

                    # Predict on the test set
                    y_test_pred = model.predict(X_test)
                    test_accuracy = accuracy_score(y_test, y_test_pred)

                    # print("Test Accuracy: ", test_accuracy)

                    # Cross-validation for accuracy
                    accuracies = cross_val_score(model, x, y_encoded, cv=cv, scoring='accuracy')
                    mean_accuracy = accuracies.mean()

                    # Cross-validation for AUC
                    y_pred_proba = cross_val_predict(model, x, y_encoded, cv=cv, method='predict_proba')

                    lb = LabelBinarizer()
                    y_binarized = lb.fit_transform(y_encoded)
                    if y_binarized.shape[1] == 1:
                        auc = roc_auc_score(y_binarized, y_pred_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_binarized, y_pred_proba, multi_class='ovr')

                    # Store the result
                    results.append((method, model_name, model, mean_accuracy, auc))
                except Exception as e:
                    print(f"Failed to evaluate model {model_name} on method {method} due to error: {e}")

        # Sort the results by accuracy in descending order and select the top 5
        results.sort(key=lambda x: x[3], reverse=True)

        results_df = pd.DataFrame(results, columns=['Method', 'Model', 'Model_Object', 'Accuracy', 'AUC'])

        unique_results = self.extract_top_unique_methods(results_df, 'Method', 3)

        # Assuming unique_results is your DataFrame with 'Method' and 'Model' columns
        unique_results_list = unique_results.apply(lambda row: f"{row['Method']}_{row['Model']}", axis=1).tolist()

        # Creating plot showing model performance
        self.plot_model_performance(results, unique_results_list)

        # Colors for different models
        colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'])

        plt.figure(figsize=(10, 8))
        for (_, row), color in zip(unique_results.iterrows(), colors):
            method = row['Method']
            model_name = row['Model']
            model = row['Model_Object']
            method_name = method.split("/")[-1].replace(".csv", "")
            
            data_ml = pd.read_csv(method)
            data_ml = data_ml.fillna(0)
            if 'Unnamed: 0' in data_ml.columns:
                data_ml = data_ml.drop(columns=['Unnamed: 0'])
            
            y = data_ml['Condition']
            x = data_ml.drop(columns=['Condition', 'Participant_time'])
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Assuming self.plot_roc_curve properly handles the input and plotting
            self.plot_roc_curve(model, x, y_encoded, cv, f'{model_name} ({method_name})', color)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.connectivity_models, "AUC_curve.pdf"), dpi=900)
        plt.close()

        # Feature Importance Analysis
        feature_importance_results = []

        # Only use the unique top 3 methods for feature importance analysis
        for _, row in unique_results.iterrows():
            method = row['Method']
            model_name = row['Model']
            model = row['Model_Object']

            # Reload the method data
            data_ml = pd.read_csv(method)

            # Replace remaining NaNs with 0
            data_ml = data_ml.fillna(0)

            if 'Unnamed: 0' in data_ml.columns:
                data_ml = data_ml.drop(columns=['Unnamed: 0'])

            y = data_ml['Condition']
            x = data_ml.drop(columns=['Condition', 'Participant_time'])

            # Encode target labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Retrain the model on the entire dataset
            model.fit(x, y_encoded)
            
            if model_name in ['RandomForest']:
                importance = model.feature_importances_
            elif model_name in ['LogisticRegression']:
                importance = model.coef_[0]
            elif model_name in ['SVM_Linear']:
                importance = model.coef_[0]
            else:
                # For SVM with RBF kernel, feature importance is not straightforward
                continue
            
            feature_importance_results.append((method, model_name, importance))

        # Display Feature Importance and visualize interactions
        for method, model_name, importance in feature_importance_results:
            method_name = method.split("/")[-1].replace(".csv", "")
            print(method_name)
            print(f"\nFeature importances for {model_name} on {method}:")
            importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            print(importance_df.head(10))  # Display top 10 features

            # Assuming that the features are named in a way that reflects channel interactions
            n_channels = len(self.channel_names)
            interaction_matrix = pd.DataFrame(index=self.channel_names, columns=self.channel_names, data=np.nan)

            for feature, imp in zip(importance_df['Feature'], importance_df['Importance']):
                index = int(feature)
                row, col = self.map_index_to_channels(index, n_channels)
                if row > col:  # Only fill the lower triangle
                    interaction_matrix.iloc[row, col] = imp

            # Mask for the upper triangle
            mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))

            vmin = interaction_matrix.values[~mask].min() if not np.all(mask) else None
            vmax = interaction_matrix.values[~mask].max() if not np.all(mask) else None

            if vmin == vmax == 0:
                print("Warning: Interaction matrix is all zeros.")
                vmin, vmax = None, None  # Let seaborn handle the scaling

            plt.figure(figsize=(15, 12))
            sns.heatmap(interaction_matrix, mask=mask, cmap='coolwarm', vmin=vmin, vmax=vmax, annot=False, fmt='.2f', cbar=True,
                        xticklabels=True, yticklabels=True)
            plt.title(f'Channel Interaction Importances for {model_name} on {method}')

            # Save the heatmap
            path_to_right_dir = os.path.join(self.connectivity_feature_correlations, model_name)

            if not os.path.exists(path_to_right_dir):
                os.makedirs(path_to_right_dir)

            plt.savefig(os.path.join(path_to_right_dir, f"Channel_Interaction_Importances_{model_name}_on_{method_name}.pdf"))
            # plt.show()

            interaction_matrix.to_csv(os.path.join(self.connectivity_feature_importance, f"Channel_Interaction_Importances_{model_name}_on_{method_name}.csv"))

        # Save the performance of all models to a CSV file
        results_df_all = pd.DataFrame(results, columns=['Method', 'Model', 'ModelInstance', 'Accuracy', 'AUC'])
        results_df_all.to_csv(os.path.join(self.connectivity_models, 'Performance_of_all_models.csv'), index=False)

def testing_connectivity_differences(self):
    """
    Test for significant differences in connectivity between different conditions (ATX, DNP, PCB).

    This function compares the connectivity data for different conditions using t-tests and 
    identifies significant differences. The results are saved as CSV files and visualized as heatmaps.
    """
    correct_path = self.connectivity_models.replace("Test", "Train")
    
    # Load data
    df_top = pd.read_csv(os.path.join(correct_path, 'Performance_of_all_models.csv'))

    # Process method names
    df_top['Processed_Method'] = df_top['Method'].str.split('/').str[-1].str.replace(".csv", "")

    methods = df_top['Processed_Method'].to_list()

    # Get unique prefixes and find the first occurrence of each
    top_3_unique_prefixes = []
    seen_prefixes = set()
    first_rows = []  # List to store the first rows corresponding to each unique method

    for method in methods:
        method = method.replace("-sq", "")
        prefix = method.split('_')[0]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            top_3_unique_prefixes.append(method)
            # Find the first row in df_top where 'Processed_Method' matches 'method'
            first_row = df_top[df_top['Processed_Method'] == method].iloc[0]
            first_rows.append(first_row)

    # Create a new DataFrame from the first rows
    df_unique_methods = pd.DataFrame(first_rows)

    methods = df_unique_methods['Method'].to_list()
    methods_unique = set(methods)

    for method in methods_unique:
        if self.train_or_test == "Test": 
            method = method.replace("Train", "Test")
            print(method)
        try:
            df = pd.read_csv(method)
        except FileNotFoundError: 
            print(f"This file has likely not been ran for the Test set")
            continue
        except: 
            print("Different error, remove the try/except structure to see more")

        method_name = method.split("/")[-1].replace(".csv", "")
        print(f"Used Method: {method_name}")
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        conditions = df['Condition']
        features = df.drop(columns=['Condition'])

        df_pcb = df[df['Condition'] == 'PCB'].drop(columns=['Condition'])
        df_atx = df[df['Condition'] == 'ATX'].drop(columns=['Condition'])
        df_dnp = df[df['Condition'] == 'DNP'].drop(columns=['Condition'])

        # Ensure participant values are matched for the t-tests
        common_participants = set(df_pcb['Participant_time']).intersection(
            set(df_atx['Participant_time'])).intersection(set(df_dnp['Participant_time']))
        
        # Filter dataframes to include only common participants
        df_pcb = df_pcb[df_pcb['Participant_time'].isin(common_participants)]
        df_atx = df_atx[df_atx['Participant_time'].isin(common_participants)]
        df_dnp = df_dnp[df_dnp['Participant_time'].isin(common_participants)]

        # Sort the dataframes by 'Participant_time' to ensure alignment
        df_pcb = df_pcb.sort_values('Participant_time').reset_index(drop=True)
        df_atx = df_atx.sort_values('Participant_time').reset_index(drop=True)
        df_dnp = df_dnp.sort_values('Participant_time').reset_index(drop=True)

        # Ensure the dataframes have the same length and participant order
        assert len(df_pcb) == len(df_atx) == len(df_dnp)
        assert all(df_pcb['Participant_time'] == df_atx['Participant_time']) == all(df_atx['Participant_time'] == df_dnp['Participant_time'])

        # Drop the 'Participant_time' column to get the feature matrices
        features_pcb = df_pcb.drop(columns=['Participant_time']).values
        features_atx = df_atx.drop(columns=['Participant_time']).values
        features_dnp = df_dnp.drop(columns=['Participant_time']).values

        # Calculate the absolute mean for each condition
        absolute_mean_atx = np.abs(features_atx).mean()
        absolute_mean_dnp = np.abs(features_dnp).mean()
        absolute_mean_pcb = np.abs(features_pcb).mean()

        mean_atx = self.process_data(df_atx)
        mean_dnp = self.process_data(df_dnp)
        mean_pcb = self.process_data(df_pcb)

        # Perform t-tests between conditions
        t_statistic_atx_pcb, p_value_atx_pcb = ttest_rel(mean_atx, mean_pcb)
        t_statistic_dnp_pcb, p_value_dnp_pcb = ttest_rel(mean_dnp, mean_pcb)

        # Print the t-statistics and p-values
        print("T-test between ATX and PCB:")
        print("T-statistic:", t_statistic_atx_pcb, "P-value:", p_value_atx_pcb)

        print("T-test between DNP and PCB:")
        print("T-statistic:", t_statistic_dnp_pcb, "P-value:", p_value_dnp_pcb)

        # Initialize lists to store p-values
        p_values_pcb_atx = []
        p_values_pcb_dnp = []
        indices = []

        # Perform paired t-tests for the upper triangle of the connectivity matrix
        for i in range(64):
            for j in range(i+1, 64):
                index = i * 64 + j
                t_stat, p_val_atx = ttest_rel(features_pcb[:, index], features_atx[:, index])
                t_stat, p_val_dnp = ttest_rel(features_pcb[:, index], features_dnp[:, index])
                p_values_pcb_atx.append(p_val_atx)
                p_values_pcb_dnp.append(p_val_dnp)
                indices.append((i, j))

        # Convert p-values to numpy arrays
        p_values_pcb_atx = np.array(p_values_pcb_atx)
        p_values_pcb_dnp = np.array(p_values_pcb_dnp)

        # Multiple comparisons correction using FDR (Benjamini-Hochberg)
        _, p_values_pcb_atx_corrected, _, _ = multipletests(p_values_pcb_atx, alpha=0.05, method='fdr_bh')
        _, p_values_pcb_dnp_corrected, _, _ = multipletests(p_values_pcb_dnp, alpha=0.05, method='fdr_bh')

        # Initialize matrices to store corrected p-values
        p_values_pcb_atx_matrix = np.zeros((64, 64))
        p_values_pcb_dnp_matrix = np.zeros((64, 64))

        # Fill the lower triangle of the matrices with the corrected p-values
        for (i, j), p_val_atx, p_val_dnp in zip(indices, p_values_pcb_atx_corrected, p_values_pcb_dnp_corrected):
            p_values_pcb_atx_matrix[j, i] = p_val_atx
            p_values_pcb_dnp_matrix[j, i] = p_val_dnp

        # Average the connectivity matrices for ATX and DNP
        avg_atx = features_atx.mean(axis=0).reshape((64, 64))
        avg_dnp = features_dnp.mean(axis=0).reshape((64, 64))
        avg_pcb = features_pcb.mean(axis=0).reshape((64, 64))

        pd.DataFrame(avg_dnp).to_csv(os.path.join(self.connectivity_significance_testing, f"{method_name}_DNP_average.csv"))
        pd.DataFrame(avg_atx).to_csv(os.path.join(self.connectivity_significance_testing, f"{method_name}_ATX_average.csv"))
        pd.DataFrame(avg_pcb).to_csv(os.path.join(self.connectivity_significance_testing, f"{method_name}_PCB_average.csv"))

        # Create differences values
        diff_atx_pcb = features_atx - features_pcb
        diff_dnp_pcb = features_dnp - features_pcb

        avg_atx_minus_pcb = diff_atx_pcb.mean(axis=0).reshape((64,64))
        avg_dnp_minus_pcb = diff_dnp_pcb.mean(axis=0).reshape((64,64))

        pd.DataFrame(avg_atx_minus_pcb).to_csv(os.path.join(self.connectivity_significance_testing, f"{method_name}_ATX_minus_PCB_average.csv"))
        pd.DataFrame(avg_dnp_minus_pcb).to_csv(os.path.join(self.connectivity_significance_testing, f"{method_name}_DNP_minus_PCB_average.csv"))

        # Define a significance threshold
        alpha = 0.05

        # Create masks for significant p-values after correction
        significant_mask_atx = p_values_pcb_atx_matrix < alpha
        significant_mask_dnp = p_values_pcb_dnp_matrix < alpha

        # Not a good solution, not sure why they pop up currently as "True"
        np.fill_diagonal(significant_mask_atx, False)
        np.fill_diagonal(significant_mask_dnp, False)

        # Count the number of True values in each mask
        count_true_atx = np.sum(significant_mask_atx)
        count_true_dnp = np.sum(significant_mask_dnp)

        # Print the counts of True values
        print("Number of significant p-values in ATX:", count_true_atx)
        print("Number of significant p-values in DNP:", count_true_dnp)

        # Apply significance masks to the difference matrices to filter out non-significant changes
        significant_atx_diff = avg_atx_minus_pcb * significant_mask_atx
        significant_dnp_diff = avg_dnp_minus_pcb * significant_mask_dnp

        # Calculate the absolute values of the differences
        abs_atx_diff = np.abs(significant_atx_diff)
        abs_dnp_diff = np.abs(significant_dnp_diff)

        # Check where ATX or DNP connectivity is greater than PCB connectivity
        greater_atx = (abs_atx_diff > np.abs(avg_pcb)).astype(int)
        greater_dnp = (abs_dnp_diff > np.abs(avg_pcb)).astype(int)

        # Count the number of significant and positive differences where ATX or DNP is greater than PCB
        positive_significant_atx = np.sum(greater_atx)
        positive_significant_dnp = np.sum(greater_dnp)

        # Print out the results
        print("Number of significant and positive mean differences where ATX is greater than PCB:", positive_significant_atx)
        print("Number of significant and positive mean differences where DNP is greater than PCB:", positive_significant_dnp)

        path_to_right_dir = os.path.join(self.connectivity_significant_differences, method_name)

        if not os.path.exists(path_to_right_dir):
            os.makedirs(path_to_right_dir)

        self.save_high_quality_heatmap(avg_atx, f'{method_name}_Average_Connectivity-ATX', os.path.join(path_to_right_dir, f"{method_name}_Average_Connectivity-ATX.pdf"),
                                cmap='viridis', dpi=900)

        self.save_high_quality_heatmap(avg_dnp, f'{method_name}_Average_Connectivity-DNP', os.path.join(path_to_right_dir, f"{method_name}_Average_Connectivity-DNP.pdf"),
                                cmap='viridis', dpi=900)

        self.save_high_quality_heatmap(avg_pcb, f'{method_name}_Average_Connectivity-PCB', os.path.join(path_to_right_dir, f"{method_name}_Average_Connectivity-PCB.pdf"),
                                cmap='viridis', dpi=900)
        
        self.plot_heatmap_with_significance(avg_atx_minus_pcb, significant_mask_atx, method_name, f'{method_name}_Average_Connectivity_ATX_minus_PCB_(Significant_Differences_to_PCB_Highlighted)')
        self.plot_heatmap_with_significance(avg_dnp_minus_pcb, significant_mask_dnp, method_name, f'{method_name}_Average_Connectivity_DNP_minus_PCB_(Significant_Differences_to_PCB_Highlighted)')

def correlation_between_feature_and_difference(self):
    """
    Calculate the correlation between feature importance and connectivity differences.

    This function loads feature importance data and connectivity difference data,
    and calculates the correlation between them. The results are printed out.
    """
    feature_csv = glob.glob(os.path.join(self.connectivity_feature_importance, "*.csv"))
    difference_csv = glob.glob(os.path.join(self.connectivity_significance_testing, "*.csv"))

    for feature in feature_csv:
        feature_df = pd.read_csv(feature)
        method_fea_string_one = feature.split("_")[-1:][0]
        method_fea_string_one = method_fea_string_one.replace(".csv", "")

        method_fea_two = feature.split("_")[-2:]
        method_string_fea_two = "_".join(method_fea_two).replace(".csv", "")

        for difference in difference_csv:
            method_dif_one = difference.split("/")[-1].split("_")[:1][0]
            method_dif_one = method_dif_one.replace(".csv", "")

            method_dif_two = difference.split("/")[-1].split("_")[:2]
            method_string_dif_two = "_".join(method_dif_two)

            if method_dif_one == method_fea_string_one or method_string_fea_two == method_string_dif_two:
                method_df = pd.read_csv(difference)

                # Correlate the dataframes after trimming
                correlation = self.correlate_dataframes(feature_df, method_df, os.path.basename(feature), os.path.basename(difference))
                print(f"Correlation between {os.path.basename(feature)} and {os.path.basename(difference)}: {correlation}")

                        
def normal_mvpa(self): 
    """
    Perform Multivoxel Pattern Analysis (MVPA) on EEG data using various machine learning models.

    This function loads EEG data, processes it, pads it to the maximum length, and evaluates different 
    machine learning models using cross-validation. The results are saved to a CSV file.
    """
    raise ValueError("THIS CODE HAS NOT BEEN USED AND WAS WRITTEN IN LIKE 20 MINUTES, CHECK AND MAKE SURE IT WORKS")
    
    eeg_files = glob.glob(os.path.join(self.base_dir, "PROC", "Train_Data", "*", '**', '*.fif'), recursive=True)

    full_data = []
    full_labels = []

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=5000, random_state=42),
        'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
        # 'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42)
    }

    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Determine the maximum length of the files
    max_length = 0
    for file in eeg_files:
        raw = mne.io.read_raw_fif(file, preload=True)
        raw.pick_types(meg=False, eeg=True, stim=False, eog=False)
        data = raw.get_data()
        if data.shape[1] > max_length:
            max_length = data.shape[1]

    # Results storage
    results = []

    # Load, process, and pad each EEG file
    for file in eeg_files:
        raw = mne.io.read_raw_fif(file, preload=True)
        raw.pick_types(meg=False, eeg=True, stim=False, eog=False)
        raw.resample(128)
        data = raw.get_data()

        # Pad the data to the maximum length
        if data.shape[1] < max_length:
            pad_width = max_length - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), 'constant')

        # Flatten the data (channels x time points -> single vector)
        data_flattened = data.flatten()

        condition = file.split("/")[-1].split("_")[1]  # Assuming condition label is part of the filename
        labels = np.full((1,), condition)  # Ensure labels have the correct shape

        full_data.append(data_flattened)
        full_labels.append(labels)

    # Combine all data and labels
    all_data = np.vstack(full_data)
    all_labels = np.concatenate(full_labels)

    print(all_data.shape)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_data)

    # Evaluate each model
    for model_name, model in models.items():
        print(f"Evaluating {model_name}")
        
        # Cross-validation for accuracy
        accuracies = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy', verbose=2)
        mean_accuracy = accuracies.mean()
    
        # Store the result
        results.append((model_name, mean_accuracy))
        print(f"{model_name} - Accuracy: {mean_accuracy:.2f}")
        
    # Print final results
    for result in results:
        print(f"Model: {result[0]}, Accuracy: {result[1]:.2f}")

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    results_df.to_csv(os.path.join(self.connectivity_models, "Normal_MVPA_results.csv"), index=False)

def feature_investigation(self):
    """
    Investigate the features by calculating correlations between different feature importance datasets.

    This function loads feature importance data, calculates correlations between different methods,
    and prints the results.
    """
    files = glob.glob(os.path.join(self.connectivity_feature_importance, "*.csv"))

    data_dict = {}

    # Load data into dictionary
    for file in files: 
        method = file.split("/")[-1].split("_")[-2:]
        method_string = "_".join(method).replace(".csv", "")
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]  # Exclude the first column
        data_dict[file] = df

    print("Running Correlations")

    correlation_results = []

    for file1, method1 in data_dict.items():
        for file2, method2 in data_dict.items():
            if file1 != file2:
                # Compute the correlation coefficient between the two datasets
                # Flatten both datasets into 1D arrays and compute the Pearson correlation
                r_value = np.corrcoef(method1.values.flatten(), method2.values.flatten())[0, 1]
                file1 = file1.replace("/home/c12172812/RS-Copy/Stijn/Results/Connectivity/Models/Feature_Importance/Channel_Interaction_Importances_", "")
                file2 = file2.replace("/home/c12172812/RS-Copy/Stijn/Results/Connectivity/Models/Feature_Importance/Channel_Interaction_Importances_", "")

                correlation_results.append((file1, file2, r_value))

    correlation_results.sort(key=lambda x: x[2], reverse=True)

    # Print the correlation results
    for result in correlation_results:
        file1, file2, r_value = result
        print(f"Correlation between {file1} and {file2}: {r_value:.4f}")

def testing_on_new_data(self):
    """
    Test machine learning models on new (test) data.

    This function loads the best-performing models from training, applies them to test data, 
    and evaluates their performance. The results are saved to a CSV file.
    """

    # Ensure that the code is not run with training data
    if "Train" in self.connectivity_models:
        raise ValueError("You are currently running this code with 'Train' instead of 'Test', change this in the MainRest please")

    correct_path = self.connectivity_models.replace("Test", "Train")

    # Load data
    df_top = pd.read_csv(os.path.join(correct_path, 'Performance_of_all_models.csv'))

    # Create a new DataFrame from the first rows
    df_unique_methods = self.extract_top_unique_methods(df_top, 'Method', 5)

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=5000, random_state=42),
        'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42)
    }

    # Set up for the correlation comparison
    model_performance = []

    if self.train_or_test == "Test":
        path_to_train = self.connectivity_overall.replace("Test", "Train")
        dfs_train = glob.glob(os.path.join(path_to_train, "*.csv"))
        dfs_test = glob.glob(os.path.join(self.connectivity_overall, "*.csv"))
        path_to_best_df = self.connectivity_models.replace("Test", "Train")

    elif self.train_or_test == "Train":
        path_to_test = self.connectivity_overall.replace("Test", "Train")
        dfs_train = glob.glob(os.path.join(self.connectivity_overall, "*.csv"))
        dfs_test = glob.glob(os.path.join(path_to_test, "*csv"))

    for i, row in df_unique_methods.iterrows():
        data_ml_train = pd.read_csv(row['Method'])
        model_name = row['ModelInstance']

        # Select the appropriate model
        if "Logistic" in model_name:
            model = models['LogisticRegression']
        elif "Random" in model_name:
            model = models['RandomForest']
        elif "linear" in model_name:
            model = models['SVM_Linear']
        elif "rbf" in model_name:
            model = models['SVM_RBF']

        # Replace remaining NaNs with 0
        data_ml_train = data_ml_train.fillna(0)

        if data_ml_train.shape[0] < 100:
            continue

        if 'Unnamed: 0' in data_ml_train.columns:
            data_ml_train = data_ml_train.drop(columns=['Unnamed: 0'])

        y_train = data_ml_train['Condition']
        x_train = data_ml_train.drop(columns=['Condition', 'Participant_time'])

        file_for_test = row['Method'].replace("Train", "Test")

        try:
            data_ml_test = pd.read_csv(file_for_test)
        except FileNotFoundError:
            print(f"Method has not been run yet for Test Data: {row['Method']}")
            continue

        # Replace remaining NaNs with 10
        data_ml_test = data_ml_test.fillna(10)

        if data_ml_test.shape[0] < 100:
            continue

        if 'Unnamed: 0' in data_ml_test.columns:
            data_ml_test = data_ml_test.drop(columns=['Unnamed: 0'])

        y_test = data_ml_test['Condition']
        x_test = data_ml_test.drop(columns=['Condition', 'Participant_time'])

        # Encode target labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.fit_transform(y_test)

        # Train the model on training data
        model.fit(x_train, y_train)

        # Evaluate the model on training data
        y_train_pred = model.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Evaluate the model on test data
        y_test_pred = model.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(row['Method'])
        print(test_accuracy)

        model_performance.append({"Method": row['Method'], "Model": model_name, "Accuracy": test_accuracy})

    performance_df = pd.DataFrame(model_performance)

    # Save the performance results to a CSV file
    performance_df.to_csv(os.path.join(self.connectivity_models, "Performance_on_Test.csv"))

def correlate_differences_test_and_train(self):
    """
    Correlate differences between test and train datasets for connectivity significance testing.

    This function loads the train and test datasets, matches files between them, 
    computes the correlation between the lower triangular parts of the matrices, 
    and saves the results and scatter plots.
    """
    
    # Determine paths for train and test data based on the current mode (train or test)
    if self.train_or_test == "Test":
        path_to_train = re.sub(r'\bTest\b', 'Train', self.connectivity_significance_testing)
        dfs_train = glob.glob(os.path.join(path_to_train, "*csv"))
        dfs_test = glob.glob(os.path.join(self.connectivity_significance_testing, "*.csv"))

    if self.train_or_test == "Train":
        path_to_test = self.connectivity_significance_testing.replace("Train", "Test")
        dfs_test = glob.glob(os.path.join(path_to_test, "*csv"))
        dfs_train = glob.glob(os.path.join(self.connectivity_significance_testing, "*.csv"))

    results = []

    # Loop through train and test files to find matches and compute correlations
    for file_train in dfs_train:
        for file_test in dfs_test:
            train_end = file_train.split("/")[-1]
            test_end = file_test.split("/")[-1]

            if train_end == test_end:
                # Load train and test data, removing the first column
                df_train = pd.read_csv(file_train).iloc[:, 1:]
                df_test = pd.read_csv(file_test).iloc[:, 1:]

                # Check for NaN values and skip files with NaNs
                if df_train.isnull().values.any() or df_test.isnull().values.any():
                    print(f"Skipping {train_end} due to NaN values.")
                    continue

                # Get the lower triangular indices of the matrix
                tril_indices = np.tril_indices_from(df_train, k=-1)

                # Flatten the lower triangular parts of the matrices
                flattened_train = df_train.values[tril_indices].flatten()
                flattened_test = df_test.values[tril_indices].flatten()

                # Compute the correlation between the flattened matrices
                correlation, p_value = pearsonr(flattened_train, flattened_test)

                print(f"Correlation between {train_end} train and test data: {correlation}")
                print(f"P-value: {p_value}")

                results.append({
                    'File': train_end,
                    'Correlation': correlation,
                    'P-value': p_value
                })

                train_end = train_end.replace(".csv", "")

                # Define path for saving correlation plots
                path_for_correlation_train_test = os.path.join(self.connectivity_graphs, "Correlations_Train_Test")

                if not os.path.exists(path_for_correlation_train_test):
                    os.makedirs(path_for_correlation_train_test)

                # Create and save scatter plot for the correlation
                plt.figure(figsize=(8, 6))
                plt.scatter(flattened_train, flattened_test, alpha=0.5)
                plt.title(f'Scatterplot of {train_end} Train vs Test Data\nCorrelation: {correlation:.2f}, P-value: {p_value:.2e}')
                plt.xlabel('Train Data')
                plt.ylabel('Test Data')
                plt.grid(True)
                plt.savefig(os.path.join(path_for_correlation_train_test, f"Correlation_for_{train_end}.pdf"))
                plt.close()

    # Create a DataFrame from the results and save it as a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(self.connectivity_significance_testing, 'Correlation_Between_Train_and_Test.csv'), index=False)

"""Helper functions below"""

def plot_model_performance(self, results, top_3_method_names):
    """
    Plot the model performance, highlighting the top 3 unique methods in a different color.
    
    Parameters:
    - results (list): List of tuples containing method, model, model object, accuracy, and AUC.
    - top_3_method_names (list): List of top 3 unique method names.
    """
    # Process results for plotting
    results = [(method.split("/")[-1].replace(".csv", ""), model, model_obj, acc, auc) for method, model, model_obj, acc, auc in results]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Method', 'Model', 'Model_Object', 'Accuracy', 'AUC'])

    # Sort by accuracy
    sorted_results = results_df.sort_values(by='Accuracy', ascending=False)

    # Create combined method-model names for comparison
    combined_names = [f"{row['Method']}_{row['Model']}" for index, row in sorted_results.iterrows()]

    # Adjust top_3_method_names to strip paths and extensions if necessary
    top_3_method_names = [name.split("/")[-1].replace(".csv", "") for name in top_3_method_names]

    # Create a list for x-tick labels and colors
    labels = [f"{row['Method']} ({row['Model']})" for index, row in sorted_results.iterrows()]
    colors = ['green' if combined_name in top_3_method_names else 'gray' for combined_name in combined_names]

    # Plotting
    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(sorted_results)), sorted_results['Accuracy'], color=colors)
    
    plt.axhline(y=0.33, color='black', linestyle='--', label='Chance (33%)')

    plt.xlabel('Method (Model)')
    plt.ylabel('Accuracy')
    plt.title('Model Performance')

    plt.legend(handles=[bars[0]], labels=['Top 3 Unique Methods'], loc='upper right')

    plt.savefig(os.path.join(self.connectivity_models, "Model_performance.pdf"), dpi=900)
    plt.close()

def plot_roc_curve(self, model, x, y, cv, label, color):
        """
        Plot ROC curve for the given model.

        Parameters:
        - model (sklearn model): The machine learning model to evaluate.
        - x (array-like): Feature matrix.
        - y (array-like): Target vector.
        - cv (cross-validation generator): Cross-validation splitting strategy.
        - label (str): Label for the plot.
        - color (str): Color for the ROC curve.
        """
        y_pred_proba = cross_val_predict(model, x, y, cv=cv, method='predict_proba')
        lb = LabelBinarizer()
        y_binarized = lb.fit_transform(y)
        
        if y_binarized.shape[1] == 1:
            fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})', color=color)
        else:
            # Initialize dictionaries for multi-class ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # Compute ROC curve and ROC area for each class
            for i in range(y_binarized.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute macro-average ROC curve and ROC area
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_binarized.shape[1])]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(y_binarized.shape[1]):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= y_binarized.shape[1]
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot macro-average ROC curve
            plt.plot(fpr["macro"], tpr["macro"], lw=2, label=f'{label} (AUC = {roc_auc["macro"]:.4f})', color=color)

def plot_heatmap_with_significance(self, avg_matrix, significant_mask, method, title):
        """
        Plot heatmap with significance markers.

        Parameters:
        - avg_matrix (np.array): Matrix of average values to plot.
        - significant_mask (np.array): Boolean mask indicating significant values.
        - method (str): Method name to be included in the plot title.
        - title (str): Title of the plot.
        """
        mask = np.triu(np.ones_like(avg_matrix, dtype=bool))  # Mask for the upper triangle
        plt.figure(figsize=(12, 10), dpi=900)
        sns.heatmap(avg_matrix, mask=mask, cmap='viridis', cbar=True, xticklabels=self.channel_names, yticklabels=self.channel_names)
        plt.title(title)
        plt.xlabel('Channel')
        plt.ylabel('Channel')

        # Overlay significance boxes
        for i in range(significant_mask.shape[0]):
            for j in range(significant_mask.shape[1]):
                if significant_mask[i, j] and i > j:  # Only mark the lower triangle
                    plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(os.path.join(self.connectivity_significant_differences, method, title + ".pdf"), dpi=900)
        plt.close()

def cluster_significant_features(self, significant_mask, title, file_path):
        """
        Cluster significant features and plot the dendrogram.

        Parameters:
        - significant_mask (np.array): Boolean mask indicating significant values.
        - title (str): Title of the plot.
        - file_path (str): Path to save the plot.
        """
        # Extract the coordinates of significant features
        significant_coords = np.argwhere(significant_mask)
        if len(significant_coords) > 0:
            # Perform hierarchical clustering
            linkage_matrix = linkage(significant_coords, method='ward')
            plt.figure(figsize=(10, 8))

            # Plot dendrogram with limited levels
            dendrogram(
                linkage_matrix, 
                labels=[f'({x},{y})' for x, y in significant_coords], 
                truncate_mode='lastp',  # Truncate the dendrogram
                p=30  # Number of levels to show
            )

            plt.title(f'Clustering of Significant Differences - {title}')
            plt.xlabel('Channel Pairs')
            plt.ylabel('Distance')
            plt.savefig(file_path)
            plt.close()

def save_high_quality_heatmap(self, matrix, title, save_path, cmap='viridis', dpi=900):
        """
        Save a high-quality heatmap with the given parameters.

        Parameters:
        - matrix (np.array): The matrix to be visualized in the heatmap.
        - title (str): The title of the heatmap.
        - save_path (str): The path of the file to save the heatmap.
        - cmap (str): The color map to use for the heatmap (default is 'viridis').
        - dpi (int): The resolution of the saved image (default is 900).
        """
        mask = np.triu(np.ones_like(matrix, dtype=bool))  # Mask for the upper triangle
        plt.figure(figsize=(12, 10), dpi=dpi)
        sns.heatmap(matrix, mask=mask, cmap=cmap, cbar=True, xticklabels=self.channel_names, yticklabels=self.channel_names)
        plt.title(title)
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        plt.savefig(save_path, dpi=dpi)
        plt.close()

def map_index_to_channels(self, index, n_channels):
        """
        Map a flattened index to channel coordinates.

        Parameters:
        - index (int): The flattened index.
        - n_channels (int): The number of channels.

        Returns:
        - row (int): The row index corresponding to the channel.
        - col (int): The column index corresponding to the channel.
        """
        row = index // n_channels
        col = index % n_channels
        return row, col

def correlate_dataframes(self, df1, df2, method_1, method_2):
        """
        Correlate the remaining dataframes after removing the first row and column.

        Parameters:
        - df1 (pd.DataFrame): First dataframe.
        - df2 (pd.DataFrame): Second dataframe.
        - method_1 (str): Name of the first method.
        - method_2 (str): Name of the second method.

        Returns:
        - correlation (float): The correlation coefficient between the two dataframes.
        """
        # Remove the first row and column
        df1_trimmed = df1.iloc[1:, 1:]
        df2_trimmed = df2.iloc[1:, 1:]

        df1_trimmed = df1_trimmed.fillna(0)
        df2_trimmed = df2_trimmed.fillna(0)
    
        # Flatten the dataframes to 1D arrays for correlation
        df1_flattened = df1_trimmed.values.flatten()
        df2_flattened = df2_trimmed.values.flatten()

        if "RandomForest" in method_1:
            df1_flattened = np.abs(df1_flattened)
            df2_flattened = np.abs(df2_flattened)
        
        # Compute the correlation
        correlation = np.corrcoef(df1_flattened, df2_flattened)[0, 1]

        # Calculate the regression line
        coeffs = np.polyfit(df1_flattened, df2_flattened, 1)
        poly_eqn = np.poly1d(coeffs)
        regression_line = poly_eqn(df1_flattened)

        # Create directory for saving the correlation plots if it doesn't exist
        path_for_correlation_features_differences = os.path.join(self.connectivity_feature_correlations, "Correlations_Features_to_Differences", method_1)
        if not os.path.exists(path_for_correlation_features_differences):
            os.makedirs(path_for_correlation_features_differences)

        # Plot the scatter plot with the regression line
        plt.figure(figsize=(10, 6))
        plt.scatter(df1_flattened, df2_flattened, alpha=0.5)
        plt.plot(df1_flattened, regression_line, color='red', label=f'Regression Line (slope={coeffs[0]:.2f})')

        # Add the regression equation to the plot
        equation_text = f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}'
        plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white'))
        
        plt.title(f'Scatter plot of Flattened DataFrames\nCorrelation: {correlation:.2f}')
        plt.xlabel(method_1)
        plt.ylabel(method_2)
        plt.grid(True)
        plt.savefig(os.path.join(path_for_correlation_features_differences, f"{method_1}_{method_2}.pdf"), dpi=900)
        plt.close()

        return correlation

def load_and_process_eeg_file(self, file_path):
        """
        Load and process an EEG file.

        Parameters:
        - file_path (str): The path to the EEG file.

        Returns:
        - data (np.array): The processed EEG data.
        """
        raw = mne.io.read_raw_fif(file_path, preload=True)
        raw.pick_types(meg=False, eeg=True, stim=False, eog=False)
        data = raw.get_data().T
        return data

def process_data(self, df):
        """
        Process data for analysis.

        Parameters:
        - df (pd.DataFrame): The dataframe to process.

        Returns:
        - masked_means (np.array): The mean values of the lower triangular part of the matrices.
        """
        # Drop 'Participant_time' and take absolute values
        data = np.abs(df.drop(columns=['Participant_time']))
        
        # Reshape to 64x64 matrix
        matrix = data.values.reshape(-1, 64, 64)
        
        # Mask for lower triangle excluding the diagonal
        mask = np.tri(N=64, k=-1, dtype=bool)
        
        # Apply mask and calculate mean across the masked elements for each participant
        masked_means = np.array([m[mask].mean() for m in matrix])
        
        return masked_means

def extract_top_unique_methods(self, df, column_name, nr_to_select=3):
        """
        Extract the top unique methods based on the given column in a DataFrame.
        Ensures all original columns in the DataFrame are retained in the output.

        Parameters:
        - df (pd.DataFrame): The original DataFrame.
        - column_name (str): The name of the column to analyze for method names.
        - nr_to_select (int): The number of unique methods to return.

        Returns:
        - df_unique_methods (pd.DataFrame): A DataFrame containing only the rows of the top unique methods.
        """
        # Copy the relevant columns to avoid modifying the original DataFrame
        df_top = df.copy()
        
        # Process the column to extract the cleaned method names
        df_top['Processed_Method'] = df_top[column_name].str.split('/').str[-1].str.replace(".csv", "")
        
        # Initialize containers for tracking unique prefixes and the first occurrence of each
        seen_prefixes = set()
        first_rows = []  # List to store the first rows corresponding to each unique method

        # Iterate over each method to determine uniqueness
        for idx, row in df_top.iterrows():
            method = row['Processed_Method']
            method_clean = method.replace("-sq", "")
            prefix = method_clean.split('_')[0]
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                first_rows.append(row)
                if len(seen_prefixes) == nr_to_select:
                    break

        # Create a new DataFrame from the first rows
        df_unique_methods = pd.DataFrame(first_rows)
        return df_unique_methods