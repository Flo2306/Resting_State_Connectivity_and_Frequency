# Florian Burger yadaadadaada
import sys
import os

sys.path.append("/usr/local/fsl/lib/python3.10/site-packages")

import mne
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import numpy as np
import time
import scipy.stats as stats
import pingouin as pg
import plotly.graph_objects as go

from scipy.stats import ttest_rel, pearsonr
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mne.time_frequency import tfr_array_morlet
from statsmodels.stats.anova import AnovaRM
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

class Analysis_individual(object):
    def __init__(self, base_dir, train_or_test, ID=None):
        """
        Initialize the Analysis_individual class.

        Parameters:
        - base_dir (str): Base directory path.
        - train_or_test (str): Specify whether the data is for training or testing.
        - ID (str): Identifier for the individual.
        """
        self.ID_full = ID
        self.ID = self.ID_full.split("_")[0]
        self.base_dir = base_dir
        self.train_or_test = train_or_test
        self.sam_or_stijn = os.path.basename(self.base_dir)

        self.train_folder = os.path.join(self.base_dir, "PROC", "Train_Data", self.ID)
        self.test_folder = os.path.join(self.base_dir, "PROC", "Test_Data", self.ID)

        if self.train_or_test == "Train":
            self.spectral_results = os.path.join(self.base_dir, "Results", "Train", "Spectral", self.ID)
            self.pupil_results = os.path.join(self.base_dir, "Results", "Train", "Pupillometry", self.ID)

        if self.train_or_test == "Test":
            self.spectral_results = os.path.join(self.base_dir, "Results", "Test", "Spectral", self.ID)
            self.pupil_results = os.path.join(self.base_dir, "Results", "Test", "Pupillometry", self.ID)
        
        # Create necessary directories if they don't exist
        for folder in [self.train_folder, self.test_folder, self.spectral_results, self.pupil_results]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        plt.style.use('seaborn-v0_8-bright')

    def train_test_split(self, train_ratio, test_ratio):
        """
        Split the EEG data into training and testing sets based on specified ratios.

        Parameters:
        - train_ratio (float): Ratio of the data to be used for training.
        - test_ratio (float): Ratio of the data to be used for testing.
        """
        if train_ratio == 0 or test_ratio == 0:
            raise ValueError("Please do not enter 0 for either ratio as there will be small calculation errors with 0 values")
        
        eeg_folder = os.path.join(self.base_dir, "PROC", "Combined", self.ID) 
        eeg_files = glob.glob(os.path.join(eeg_folder, '**', '*.fif'), recursive=True)

        excel_file = os.path.join(self.base_dir, "drug_order.xlsx")
        df = pd.read_excel(excel_file)

        for file_path in eeg_files:
            filename = os.path.basename(file_path)
            subject = int(filename.split('_')[0])

            if self.sam_or_stijn == "Samuel":
                task = filename.split('_')[1].replace("RS", "")
                
                # Find condition based on subject ID in the Excel file
                placebo_condition = df.loc[df['Subject'] == int(subject), 'Placebo'].values[0]

                if int(task) == placebo_condition:
                    condition = "PCB"
                else:
                    condition = "MEM"
            else:
                task = filename.split('_')[1]
                drug_col = f'Drug_{task}'
                subject_row = df[df['Subject'] == subject]
                condition = subject_row[drug_col].values[0]

            raw = mne.io.read_raw_fif(file_path, preload=True)
            raw.interpolate_bads(reset_bads=True, mode='accurate')

            # Determine the indices for train and test data
            n_samples = len(raw)
            sfreq = raw.info['sfreq']
            train_end_time = int(train_ratio * n_samples) / sfreq
            test_start_time = int((1 - test_ratio) * n_samples) / sfreq

            if int(self.ID) % 2 == 0:  # If subject ID is even
                train_raw = raw.copy().crop(0, train_end_time)
                test_raw = raw.copy().crop(train_end_time, ((n_samples - 1) / sfreq) - 0.05)
            else:  # If subject ID is odd
                test_end_time = int(test_ratio * n_samples) / sfreq
                train_raw = raw.copy().crop(test_end_time, ((n_samples - 1) / sfreq))
                test_raw = raw.copy().crop(0, test_end_time)

            # Write train and test data to separate files
            train_file = os.path.join(self.train_folder, f"{subject}_{condition}_train_raw.fif")
            test_file = os.path.join(self.test_folder, f"{subject}_{condition}_test_raw.fif")

            train_raw.save(train_file, overwrite=True)
            test_raw.save(test_file, overwrite=True)

    def calculating_log_spectral(self, input_file, condition):
        """
        Calculate the logarithmic spectral power for the given input files.

        Parameters:
        - input_file (list): List of PSD arrays.
        - condition (str): The condition label.

        Returns:
        - mean_centered_log_power (np.array): The mean centered logarithmic power.
        """
        psd_arrays = np.array(input_file)
        mean_psd = np.mean(psd_arrays, axis=0)
        power = mean_psd ** 2
        log_power = np.log(power)
        mean_centered_log_power = log_power - np.mean(log_power)

        np.savetxt(os.path.join(self.spectral_results, f"Frequency_array_{condition}.csv"), mean_centered_log_power, delimiter=",")
        return mean_centered_log_power

    def create_plot_time_frequency(self, data_dict, condition, sfreq, frequencies):
        """
        Create a time-frequency plot and save it.

        Parameters:
        - data_dict (dict): Dictionary of data arrays.
        - condition (str): The condition label.
        - sfreq (float): Sampling frequency.
        - frequencies (array-like): Array of frequencies.
        """
        all_data = np.stack(list(data_dict.values()))
        mean_data = np.mean(all_data, axis=0)
        final_mean_array = np.mean(mean_data, axis=0)

        np.savetxt(os.path.join(os.path.dirname(self.spectral_results), f"Time_frequency_{condition}.csv"), final_mean_array, delimiter=",")

        plt.figure(figsize=(10, 6))
        plt.imshow(20 * np.log10(final_mean_array), aspect='auto', origin='lower',
                   extent=[0, final_mean_array.shape[-1] / sfreq, frequencies[0], frequencies[-1]],
                   cmap='viridis')
        plt.colorbar(label='Power (dB)')
        plt.title('Mean Time-Frequency Representation Across Subjects')
        plt.xlabel('Time (indices)')
        plt.ylabel('Frequency (indices)')
        plt.savefig(os.path.join(os.path.dirname(self.spectral_results), f"Time_frequency_{condition}.pdf"))

    def spectral_analysis_per_subject(self):
        """
        Perform spectral analysis for each subject.

        This function processes the EEG data files, computes the power spectral density (PSD),
        and the time-frequency representation using Morlet wavelets.
        """
        if self.train_or_test == "Train":
            eeg_files = glob.glob(os.path.join(self.train_folder, '**', '*.fif'), recursive=True)
        else:
            eeg_files = glob.glob(os.path.join(self.test_folder, '**', '*.fif'), recursive=True)

        psd_arrays_control = []
        psd_arrays_experimental_mem = []
        psd_arrays_experimental_dnp = []
        psd_arrays_experimental_atx = []

        time_frequency_control = {}
        time_frequency_mem = {}
        time_frequency_dnp = {}
        time_frequency_atx = {}

        for i, file_path in enumerate(eeg_files):
            filename = os.path.basename(file_path)
            subject = filename.split('_')[0]
            condition = filename.split('_')[1]

            raw = mne.io.read_raw_fif(file_path, preload=True)
            picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, ecog=False)
            raw.pick_channels([raw.ch_names[i] for i in picks_eeg])

            # Compute PSD using Welch's method
            psd, freqs = mne.time_frequency.psd_welch(raw, fmin=0, fmax=49, average='mean', n_fft=2048)
            mean_psd = np.mean(psd, axis=0)  # Averaging across channels if not already averaged

            # Compute time-frequency using Morlet wavelets
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            frequencies = np.logspace(np.log10(1), np.log10(50), num=50)
            n_cycles = frequencies / 2.0
            epoch_length = 30  # seconds
            sample_length = int(sfreq * epoch_length)
            n_epochs = data.shape[1] // sample_length
            epochs_data = np.array([data[:, i * sample_length:(i + 1) * sample_length] for i in range(n_epochs)])
            power = tfr_array_morlet(epochs_data, sfreq=sfreq, freqs=frequencies, n_cycles=n_cycles, output='power', verbose=3)
            mean_time = power.mean(axis=1)

            if condition == "PCB":
                psd_arrays_control.append(mean_psd)
                time_frequency_control[subject] = mean_time
            elif condition == "MEM":
                psd_arrays_experimental_mem.append(mean_psd)
                time_frequency_mem[subject] = mean_time
            elif condition == "ATX":
                psd_arrays_experimental_atx.append(mean_psd)
                time_frequency_atx[subject] = mean_time
            elif condition == "DNP":
                psd_arrays_experimental_dnp.append(mean_psd)
                time_frequency_dnp[subject] = mean_time

        # Calculate and plot mean centered log power for control condition
        mean_centered_log_power_control = self.calculating_log_spectral(psd_arrays_control, "control")
        plt.figure()
        plt.plot(freqs, mean_centered_log_power_control, color='blue', label='Control')
        self.create_plot_time_frequency(time_frequency_control, "PCB", sfreq=sfreq, frequencies=frequencies)

        # Adjust plot based on the type of data (Stijn or Samuel)
        if self.sam_or_stijn == "Stijn":
            mean_centered_log_power_experimental_atx = self.calculating_log_spectral(psd_arrays_experimental_atx, "atx")
            mean_centered_log_power_experimental_dnp = self.calculating_log_spectral(psd_arrays_experimental_dnp, "dnp")
            plt.plot(freqs, mean_centered_log_power_experimental_dnp, color='green', label='DNP')
            plt.plot(freqs, mean_centered_log_power_experimental_atx, color='red', label='ATX')
            self.create_plot_time_frequency(time_frequency_atx, "ATX", sfreq=sfreq, frequencies=frequencies)
            self.create_plot_time_frequency(time_frequency_dnp, "DNP", sfreq=sfreq, frequencies=frequencies)
        else:
            mean_centered_log_power_experimental_mem = self.calculating_log_spectral(psd_arrays_experimental_mem, "mem")
            plt.plot(freqs, mean_centered_log_power_experimental_mem, color='red', label='MEM')
            self.create_plot_time_frequency(time_frequency_mem, "MEM", sfreq=sfreq, frequencies=frequencies)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Log(Power(V^2/Hz))')
        plt.title('Average Power Spectral Density across Files')
        plt.legend()
        plt.savefig(os.path.join(self.spectral_results, f"Overall_Effect_{self.train_or_test}.pdf"))

        np.savetxt(os.path.join(os.path.dirname(self.spectral_results), f"Frequency_array.csv"), freqs, delimiter=",")

    def correlation_arousal_channel(self, max_shift=1, step_size=100):
        """
        Calculate the correlation between EEG and ECoG channels for different time shifts.

        Parameters:
        - max_shift (int): Maximum shift in milliseconds.
        - step_size (int): Step size for shifting in milliseconds.
        """
        if self.train_or_test == "Train":
            eeg_files = glob.glob(os.path.join(self.train_folder, '**', '*.fif'), recursive=True)
        elif self.train_or_test == "Test":
            eeg_files = glob.glob(os.path.join(self.test_folder, '**', '*.fif'), recursive=True)

        for eeg_file in eeg_files:
            raw = mne.io.read_raw_fif(eeg_file, preload=True)
            raw.interpolate_bads(reset_bads=True, mode='accurate')

            ecog_channel_index = mne.pick_types(raw.info, ecog=True, exclude='bads')[0]
            eeg_channel_indices = mne.pick_types(raw.info, eeg=True, exclude='bads')

            pupil_raw = raw.get_data(ecog_channel_index)[0]

            for shift_ms in tqdm(range(0, max_shift, step_size)):
                shift_number = int(shift_ms * raw.info['sfreq'] / 1000)
                ecog_data = pupil_raw[shift_number:]

                correlation_data = []

                for eeg_index in eeg_channel_indices:
                    eeg_channel_name = raw.ch_names[eeg_index]
                    eeg_data_raw = raw.get_data(eeg_index)[0]
                    eeg_data = eeg_data_raw[:len(eeg_data_raw) - shift_number]

                    correlation, p_value = pearsonr(eeg_data, ecog_data)
                    bf10 = pg.bayesfactor_pearson(correlation, len(eeg_data))

                    correlation_data.append({'EEG_channel_name': eeg_channel_name, 
                                            'correlation': correlation, 
                                            'p-value': p_value, 
                                            'BF10': bf10,
                                            'time_shift_ms': shift_ms})

                    model_first_order = LinearRegression()
                    model_first_order.fit(eeg_data[:, np.newaxis], ecog_data)
                    slope_first_order = model_first_order.coef_[0]
                    correlation_data[-1]['first_order_slope'] = slope_first_order

                    polynomial_features = PolynomialFeatures(degree=2)
                    x_poly = polynomial_features.fit_transform(eeg_data[:, np.newaxis])
                    model_second_order = LinearRegression()
                    model_second_order.fit(x_poly, ecog_data)
                    quadratic_coef = model_second_order.coef_[2]
                    correlation_data[-1]['second_order_quadratic_coef'] = quadratic_coef

                correlation_df = pd.DataFrame(correlation_data)
                split_list = os.path.basename(eeg_file).split("_")[:2]
                file_name_string = "_".join(split_list)
                shift_seconds = shift_ms / 1000
                file_name = f"Correlation_{file_name_string}_shift_{shift_seconds}s_df.csv"
                file_path = os.path.join(self.pupil_results, file_name)
                correlation_df.to_csv(file_path, index=False)


class Analysis_group(object):
    def __init__(self, base_dir, train_or_test, ID=None):
        """
        Initialize the Analysis_group class.

        Parameters:
        - base_dir (str): Base directory path.
        - train_or_test (str): Specify whether the data is for training or testing.
        - ID (str): Identifier for the group (optional).
        """
        self.base_dir = base_dir
        self.train_or_test = train_or_test
        self.sam_or_stijn = os.path.basename(self.base_dir)

        self.train_folder = os.path.join(self.base_dir, "PROC", "Train_Data")
        self.test_folder = os.path.join(self.base_dir, "PROC", "Test_Data")

        if self.train_or_test == "Train":
            self.spectral_results = os.path.join(self.base_dir, "Results", "Train", "Spectral")
            self.pupil_results = os.path.join(self.base_dir, "Results", "Train", "Pupillometry")

        if self.train_or_test == "Test":
            self.spectral_results = os.path.join(self.base_dir, "Results", "Test", "Spectral")
            self.pupil_results = os.path.join(self.base_dir, "Results", "Test", "Pupillometry")
        
        # Create directories if they don't exist
        for folder in [self.spectral_results, self.pupil_results]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        plt.style.use('seaborn-v0_8-bright')

    def comparing_frequencies_1(self, range, frequencies, df_control, df_exp1, df_exp2, name, all_p_values):
        """
        Compare frequency bands between control and experimental conditions.

        Parameters:
        - range (tuple): Frequency range to analyze.
        - frequencies (list): List of frequencies.
        - df_control (DataFrame): Control data.
        - df_exp1 (DataFrame): Experimental data 1.
        - df_exp2 (DataFrame): Experimental data 2.
        - name (str): Name of the frequency band.
        - all_p_values (list): List to store p-values for FDR correction.

        Returns:
        - highest_error_bar (float): Highest error bar position.
        - lowest_error_bar (float): Lowest error bar position.
        - t_stat1 (float): T-statistic for control vs. exp1.
        - p_value1 (float): P-value for control vs. exp1.
        - bayes_factor1 (float): Bayes factor for control vs. exp1.
        - t_stat2 (float): T-statistic for control vs. exp2.
        - p_value2 (float): P-value for control vs. exp2.
        - bayes_factor2 (float): Bayes factor for control vs. exp2.
        """
        # Find indices of the specified frequency range
        indices = [i for i, f in enumerate(frequencies) if range[0] <= f <= range[1]]

        # Adjust dataframes to the specified range
        df_control_adjusted = df_control.iloc[indices]
        averages_control = df_control_adjusted.mean(axis=0)
        control_mean = np.mean(averages_control)

        df_exp1_adjusted = df_exp1.iloc[indices]
        averages_exp1 = df_exp1_adjusted.mean(axis=0)
        exp1_mean = np.mean(averages_exp1)

        df_exp2_adjusted = df_exp2.iloc[indices]
        averages_exp2 = df_exp2_adjusted.mean(axis=0)
        exp2_mean = np.mean(averages_exp2)

        # Perform paired t-tests
        t_stat1, p_value1 = ttest_rel(averages_control, averages_exp1)
        t_stat2, p_value2 = ttest_rel(averages_control, averages_exp2)

        # Perform Bayesian t-tests
        results1 = pg.ttest(averages_control, averages_exp1, paired=True)
        results2 = pg.ttest(averages_control, averages_exp2, paired=True)

        bayes_factor1 = float(results1['BF10'].iloc[0])
        bayes_factor2 = float(results2['BF10'].iloc[0])

        # Calculate confidence intervals
        control_confidence = stats.t.interval(alpha=0.95, df=len(averages_control)-1, loc=np.mean(averages_control), scale=stats.sem(averages_control))
        exp1_confidence = stats.t.interval(alpha=0.95, df=len(averages_exp1)-1, loc=np.mean(averages_exp1), scale=stats.sem(averages_exp1))
        exp2_confidence = stats.t.interval(alpha=0.95, df=len(averages_exp2)-1, loc=np.mean(averages_exp2), scale=stats.sem(averages_exp2))

        # Extend all_p_values list with the new p-values
        all_p_values.extend([p_value1, p_value2])

        # Plot means and confidence intervals
        plt.errorbar(name, control_mean, yerr=[[control_mean - control_confidence[0]], [control_confidence[1] - control_mean]], fmt='o', color='blue', label='Control')
        plt.errorbar(name, exp1_mean, yerr=[[exp1_mean - exp1_confidence[0]], [exp1_confidence[1] - exp1_mean]], fmt='o', color='red', label='Atomoxetine')
        plt.errorbar(name, exp2_mean, yerr=[[exp2_mean - exp2_confidence[0]], [exp2_confidence[1] - exp2_mean]], fmt='o', color='green', label='Donepezil')

        # Calculate the highest and lowest error bar positions
        highest_error_bar = max(control_mean + (control_confidence[1] - control_mean),
                                exp1_mean + (exp1_confidence[1] - exp1_mean),
                                exp2_mean + (exp2_confidence[1] - exp2_mean))
        lowest_error_bar = min(control_mean - (control_mean - control_confidence[0]),
                               exp1_mean - (exp1_mean - exp1_confidence[0]),
                               exp2_mean - (exp2_mean - exp2_confidence[0]))

        return highest_error_bar, lowest_error_bar, t_stat1, p_value1, bayes_factor1, t_stat2, p_value2, bayes_factor2

    def spectral_across_people(self):
        """
        Perform spectral analysis across multiple subjects.

        This function processes the spectral data, compares different frequency bands, and generates plots.
        """
        csv_files = glob.glob(os.path.join(self.spectral_results, '**', '*.csv'), recursive=True)

        # Initialize empty dataframes for each category
        control_df = pd.DataFrame()
        atx_df = pd.DataFrame()
        dnp_df = pd.DataFrame()
        mem_df = pd.DataFrame()

        # Iterate over CSV files and categorize them
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if csv_file.endswith("control.csv"):
                control_df[csv_file] = df.iloc[:, 0]
            elif csv_file.endswith("atx.csv"):
                atx_df[csv_file] = df.iloc[:, 0].reset_index(drop=True)
            elif csv_file.endswith("dnp.csv"):
                dnp_df[csv_file] = df.iloc[:, 0].reset_index(drop=True)
            elif csv_file.endswith("mem.csv"):
                mem_df[csv_file] = df.iloc[:, 0].reset_index(drop=True)

        frequencies_df = pd.read_csv(os.path.join(self.spectral_results, "Frequency_array.csv"))
        frequencies = frequencies_df.iloc[:, 0].tolist()
        control_avg = control_df.mean(axis=1)

        # Define the frequency ranges for different bands
        delta_range = (0.5, 4)
        theta_range = (4, 8)
        alpha_range = (8, 12)
        beta_range = (12, 30)
        gamma_range = (30, 49)

        bands = ['Delta (0.5 - 4 Hz)', 'Theta (4 - 8 Hz)', 'Alpha (8 - 12 Hz)', 'Beta (12 - 30 Hz)', 'Gamma (30 - 49 Hz)']
        ranges = [delta_range, theta_range, alpha_range, beta_range, gamma_range]

        all_p_values = []
        t_stats = []
        bayes_factors = []

        highest_error_bars = []
        lowest_error_bars = []

        plt.figure(figsize=(10, 6))

        for i, band in enumerate(bands):
            if self.sam_or_stijn == "Samuel":
                plt.title('Means and Confidence Intervals for Different Frequency Bands for Memantine')
                high, low, t_stat1, p_value1, bayes_factor1, t_stat2, p_value2, bayes_factor2 = self.comparing_frequencies_1(ranges[i], frequencies, control_df, mem_df, mem_df, band, all_p_values)
            else:
                plt.title('Means and Confidence Intervals for Different Frequency Bands for Donepezil')
                high, low, t_stat1, p_value1, bayes_factor1, t_stat2, p_value2, bayes_factor2 = self.comparing_frequencies_1(ranges[i], frequencies, control_df, dnp_df, atx_df, band, all_p_values)

            highest_error_bars.append(high)
            lowest_error_bars.append(low)
            t_stats.extend([t_stat1, t_stat2])
            bayes_factors.extend([bayes_factor1, bayes_factor2])

        # Apply FDR correction to p-values
        reject, pvals_corrected, _, _ = multipletests(all_p_values, method='fdr_bh')

        # Annotate the plot with corrected p-values, t-statistics, and Bayes factors
        for i, band in enumerate(bands):
            highest_error_bar = highest_error_bars[i]
            lowest_error_bar = lowest_error_bars[i]

            p_val1 = pvals_corrected[i * 2]
            p_val2 = pvals_corrected[i * 2 + 1]
            t_stat1 = t_stats[i * 2]
            t_stat2 = t_stats[i * 2 + 1]
            bayes_factor1 = bayes_factors[i * 2]
            bayes_factor2 = bayes_factors[i * 2 + 1]

            plt.text(band, highest_error_bar + 0.1, f"Atomoxetine\nt-stat: {t_stat1:.2f}\np-val: {p_val1:.3f}\nBF10: {bayes_factor1:.3f}", ha='center', va='bottom', fontsize=8, color='red')
            plt.text(band, lowest_error_bar - 0.1, f"Donepezil\nt-stat: {t_stat2:.2f}\np-val: {p_val2:.3f}\nBF10: {bayes_factor2:.3f}", ha='center', va='top', fontsize=8, color='green')

        legend_labels = ['Control', 'Atomoxetine', 'Donepezil']
        legend_colors = ['blue', 'red', 'green']

        # Create custom legend handles
        legend_handles = [Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color) for color in legend_colors]

        # Plot the legend
        plt.legend(legend_handles, legend_labels)

        plt.xlabel('Frequency Band')
        plt.ylabel('Mean Log(Power(V^2/Hz))')
        plt.grid(True)
        plt.ylim(min(lowest_error_bars) - 2, max(highest_error_bars) + 2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.spectral_results, f"Means_and_Confidence_Intervals.pdf"), dpi=3000)
        plt.clf()

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, control_avg, color='blue', label='Control')

        if self.sam_or_stijn == "Stijn":
            atx_avg = atx_df.mean(axis=1)
            dnp_avg = dnp_df.mean(axis=1)
            plt.plot(frequencies, dnp_avg, color='green', label='DNP')
            plt.plot(frequencies, atx_avg, color='red', label='ATX')
        else:
            mem_avg = mem_df.mean(axis=1)
            plt.plot(frequencies, mem_avg, color='red', label='MEM')

        # Highlight different frequency bands
        plt.axvspan(0.5, 4, color='#404040', alpha=0.3, label='Delta (0.5-4 Hz)')
        plt.axvspan(4, 8, color='#606060', alpha=0.3, label='Theta (4-8 Hz)')
        plt.axvspan(8, 12, color='#8080ff', alpha=0.3, label='Alpha (8-12 Hz)')
        plt.axvspan(12, 30, color='#a0a0ff', alpha=0.3, label='Beta (12-30 Hz)')
        plt.axvspan(30, 49, color='#c0c0c0', alpha=0.3, label='Gamma (30-49 Hz)')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Log(Power(V^2/Hz))')
        plt.title('Average Power Spectral Density for Each Condition')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.spectral_results, f"Overall_Effect_{self.train_or_test}.pdf"), dpi=3000)

    def arousal_t_test(self): 
        """
        Perform t-tests and Bayesian analysis on pupil data across different conditions.
        """
        # Determine the folder based on whether the data is for training or testing
        if self.train_or_test == "Train":
            eeg_files = glob.glob(os.path.join(self.train_folder, '**', '*.fif'), recursive=True)
        elif self.train_or_test == "Test": 
            eeg_files = glob.glob(os.path.join(self.test_folder, '**', '*.fif'), recursive=True)
        
        # Initialize empty DataFrames for each condition
        control_df = pd.DataFrame()
        atx_df = pd.DataFrame()
        dnp_df = pd.DataFrame()
        mem_df = pd.DataFrame()

        # Process each EEG file
        for file_path in eeg_files: 
            filename = os.path.basename(file_path)
            subject = filename.split('_')[0]
            condition = filename.split('_')[1]

            raw = mne.io.read_raw_fif(file_path, preload=True)

            # Filter to select only Pupil (added as ECOG channels)
            picks_ecog = mne.pick_types(raw.info, meg=False, eeg=False, ecog=True)
            raw.pick_channels([raw.ch_names[i] for i in picks_ecog])

            # Extract EEG data and convert to DataFrame
            eeg_data = raw.get_data()
            eeg_data = pd.DataFrame(eeg_data.T, columns=raw.ch_names)

            # Add a 'Time' column with the time values
            eeg_data['Time'] = raw.times

            # Add subject and condition columns
            eeg_data['Subject'] = subject
            eeg_data['Condition'] = condition

            # Depending on the condition, add the EEG data to the corresponding DataFrame
            if condition == "PCB":
                control_df = pd.concat([control_df, eeg_data], ignore_index=True)
            elif condition == "ATX":
                atx_df = pd.concat([atx_df, eeg_data], ignore_index=True)
            elif condition == "DNP":
                dnp_df = pd.concat([dnp_df, eeg_data], ignore_index=True)
            elif condition == "MEM":
                mem_df = pd.concat([mem_df, eeg_data], ignore_index=True)

        combined_df = pd.DataFrame()

        # Group by 'Subject' and calculate the mean pupil size for control condition
        combined_df = control_df.groupby(['Subject'])['Pupil'].mean().reset_index()

        if self.sam_or_stijn == "Stijn":
            # Group by 'Subject' and calculate the mean pupil size for ATX and DNP conditions
            average_scores_atx = atx_df.groupby(['Subject'])['Pupil'].mean().reset_index()
            average_scores_dnp = dnp_df.groupby(['Subject'])['Pupil'].mean().reset_index()

            # Merge the dataframes
            combined_df = combined_df.merge(average_scores_atx, on='Subject', suffixes=('_control', '_atx'))
            combined_df = combined_df.merge(average_scores_dnp, on='Subject', suffixes=('_atx', '_dnp'))

            conditions = ['Pupil_atx', 'Pupil']
        else: 
            # Group by 'Subject' and calculate the mean pupil size for MEM condition
            average_scores_mem = mem_df.groupby(['Subject'])['Pupil'].mean().reset_index()

            # Merge the dataframes
            combined_df = combined_df.merge(average_scores_mem, on='Subject', suffixes=('_control', '_mem'))

            conditions = ['Pupil_mem']
        
        # Extract average scores for the control condition
        control_scores = combined_df['Pupil_control']
        print(control_scores.mean())

        for condition in conditions:
            # Extract average scores for the current condition
            condition_scores = combined_df[condition]

            print(condition_scores.mean())

            # Perform t-test
            t_statistic, p_value = ttest_rel(control_scores, condition_scores)

            # Perform Bayesian t-test
            results = pg.ttest(control_scores, condition_scores, paired=True)
            bayes_factor = float(results['BF10'].iloc[0])

            # Print results
            print(f"T-test for {condition} vs. control:")
            print("T-statistic:", t_statistic)
            print("p-value:", p_value)
            print("Bayes Factor (BF10): ", bayes_factor)

        combined_df.to_csv(os.path.join(self.pupil_results, "Average_arousal_per_subject_and_condition.csv"))

        # Plot average scores for each condition
        plt.figure(figsize=(10, 6))

        # Control condition
        sns.lineplot(data=combined_df, x='Subject', y='Pupil_control', label='Control')

        # Other conditions
        if self.sam_or_stijn == "Stijn":
            sns.lineplot(data=combined_df, x='Subject', y='Pupil_atx', label='ATX')
            sns.lineplot(data=combined_df, x='Subject', y='Pupil', label='DNP')
        else:
            sns.lineplot(data=combined_df, x='Subject', y='Pupil_mem', label='MEM')

        # Plot mean lines for each condition
        control_mean = combined_df['Pupil_control'].mean()
        plt.axhline(y=control_mean, color='blue', linestyle='--', label=f'Control Mean: {control_mean:.2f}')

        if self.sam_or_stijn == "Stijn":
            atx_mean = combined_df['Pupil_atx'].mean()
            plt.axhline(y=atx_mean, color='orange', linestyle='--', label=f'ATX Mean: {atx_mean:.2f}')

            dnp_mean = combined_df['Pupil'].mean()
            plt.axhline(y=dnp_mean, color='green', linestyle='--', label=f'DNP Mean: {dnp_mean:.2f}')
        else:
            mem_mean = combined_df['Pupil_mem'].mean()
            plt.axhline(y=mem_mean, color='red', linestyle='--', label=f'MEM Mean: {mem_mean:.2f}')

        plt.xlabel('Subject')
        plt.ylabel('Normalised Average Pupil Size')
        plt.title(f'Average Pupil Size for Different Conditions on {self.train_or_test} Set')
        plt.legend()
        plt.savefig(os.path.join(self.pupil_results, "Average_Pupil_Size.pdf"))

    def histogram_duration(self): 
        """
        Plot the histogram of EEG file durations for each condition and subject.
        """
        # Determine the folder based on whether the data is for training or testing
        if self.train_or_test == "Train":
            eeg_files = glob.glob(os.path.join(self.train_folder, '**', '*.fif'), recursive=True)
        elif self.train_or_test == "Test": 
            eeg_files = glob.glob(os.path.join(self.test_folder, '**', '*.fif'), recursive=True)
        
        file_lengths_per_subject = {}
        conditions = set()

        # Process each EEG file
        for i, file_path in enumerate(eeg_files):
            filename = os.path.basename(file_path)
            subject = filename.split('_')[0]
            condition = filename.split('_')[1]

            print(condition)

            raw = mne.io.read_raw_fif(file_path, preload=True)
            time = raw.times[-1] 

            # Store file length per subject and condition
            key = (subject, condition)
            if key not in file_lengths_per_subject:
                file_lengths_per_subject[key] = []
            file_lengths_per_subject[key].append(time)

            conditions.add(condition)

        # Extract subjects and lengths
        subjects = sorted({key[0] for key in file_lengths_per_subject.keys()})
        lengths = [[file_lengths_per_subject.get((subject, condition), []) for subject in subjects] for condition in sorted({key[1] for key in file_lengths_per_subject.keys()})]

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 8))

        bar_width = 0.2
        index = np.arange(len(subjects))

        for i, condition in enumerate(sorted({key[1] for key in file_lengths_per_subject.keys()})):
            ax.bar(index + i * bar_width, [lengths[i][j][0] if lengths[i][j] else 0 for j in range(len(subjects))], bar_width, label=condition)

        ax.set_xlabel('Subject')
        ax.set_ylabel('Length of EEG File (seconds)')
        ax.set_title('EEG File Lengths per Subject for Each Condition')
        ax.set_xticks(index + bar_width * 1.5)
        ax.set_xticklabels(subjects)
        ax.legend()

        plt.savefig(os.path.join(self.base_dir, "length_output_train.pdf"))

    def arousal_eeg_correlation(self, max_shift=1):
        """
        Analyze the correlation between EEG channels and pupil size with varying time shifts.
        """
        # Determine conditions based on the subject type
        if self.sam_or_stijn == "Stijn":
            conditions = ['PCB', 'ATX', 'DNP']
        else:
            conditions = ['PCB', 'MEM']

        combined_df = pd.DataFrame()

        # Iterate over each condition
        for condition in conditions:
            condition_df = pd.DataFrame()
            
            # Iterate over time shifts
            for shift_ms in range(0, max_shift, 100):
                df_to_combine = pd.DataFrame()
                shift_seconds = shift_ms / 1000
                csv_files = glob.glob(os.path.join(self.pupil_results, "**", f"*{condition}*{shift_seconds}*.csv"), recursive=True)
                
                for file in csv_files:
                    print(file)
                    df = pd.read_csv(file)
                    
                    if "EEG_channel_name" not in df_to_combine.columns:
                        df_to_combine['EEG_channel_name'] = df['EEG_channel_name']

                    df_to_combine[file + "_correlation"] = df['correlation']
                    df_to_combine[file + "_first_order"] = df['first_order_slope']
                    df_to_combine[file + "_second_order"] = df['second_order_quadratic_coef']

                if "EEG_channel_name" not in condition_df.columns:
                    condition_df['EEG_channel_name'] = df_to_combine['EEG_channel_name']

                # Average correlations and regression coefficients across all files for this time shift
                condition_df[str(shift_ms) + "_correlation"] = df_to_combine.filter(like='correlation').mean(axis=1)
                condition_df[str(shift_ms) + "_first_order"] = df_to_combine.filter(like='first_order').mean(axis=1)
                condition_df[str(shift_ms) + "_second_order"] = df_to_combine.filter(like='second_order').mean(axis=1)

            # Save the condition-specific dataframe to CSV
            condition_df.to_csv(os.path.join(self.pupil_results, f"Correlation_Pupil_EEG_{condition}.csv"))

            # Append condition data to the combined DataFrame
            if combined_df.empty:
                combined_df = condition_df.copy()
            else:
                combined_df = pd.merge(combined_df, condition_df, on="EEG_channel_name")

        # Setup conditions for plotting
        conditions = ['PCB', 'ATX', 'DNP']  # Adjust based on your actual conditions

        # Create a subplot for each channel and measure
        for condition in conditions:
            file_path = os.path.join(self.pupil_results, f"Correlation_Pupil_EEG_{condition}.csv")
            df = pd.read_csv(file_path)
            df.set_index('EEG_channel_name', inplace=True)

            # Determine the number of unique channels
            channels = df.index.unique()
            n_channels = len(channels)
            fig = make_subplots(rows=n_channels, cols=1, subplot_titles=[f'Channel: {ch}' for ch in channels], shared_xaxes=False,
                                specs=[[{'secondary_y': True}] for _ in range(n_channels)])  # Specify secondary y-axis for each subplot

            # Assuming time shifts are captured in the column names, e.g., '100_correlation'
            time_shifts = [int(col.split('_')[0]) for col in df.columns if 'correlation' in col]
            time_shifts = sorted(set(time_shifts))

            # Plot each measure in a loop
            measures = ['correlation', 'first_order', 'second_order']
            colors = ['blue', 'green', 'red']  # Colors for correlation, first order, second order

            for i, channel in enumerate(channels):
                for measure, color in zip(measures, colors):
                    measure_data = [df.at[channel, f"{shift}_{measure}"] for shift in time_shifts]
                    if measure == 'correlation':
                        yaxis = 'y'  # Primary y-axis for correlation
                    else:
                        yaxis = 'y2'  # Secondary y-axis for first and second order

                    fig.add_trace(
                        go.Scatter(
                            x=time_shifts,
                            y=measure_data,
                            mode='lines',
                            name=measure if i == 0 else None,
                            line=dict(color=color),
                            showlegend=i == 0,
                            hoverinfo='none',  # Disable default hover info
                            hovertemplate=f'{measure}<br>Time Shift: %{{x}} ms<br>Value: %{{y:.4f}}',
                        ),
                        row=i+1, col=1,
                        secondary_y=(yaxis == 'y2')  # Use secondary y-axis for r-squared values
                    )

            # Update layout and axes
            fig.update_layout(height=300 * n_channels, width=800, title_text=f"EEG Measures Over Time for {condition}", showlegend=True)
            fig.update_xaxes(title_text="Time Shift (ms)")
            fig.update_yaxes(title_text="Correlation Value", secondary_y=False)
            fig.update_yaxes(title_text="R-Squared Value", secondary_y=True)

            # Save the plot to HTML file
            output_html = os.path.join(self.pupil_results, f"{condition}_EEG_Plot_Interactive.html")
            fig.write_html(output_html)

        # Setup measures for combined plotting
        measures = ['correlation', 'first_order', 'second_order']
        colors = {'PCB': 'blue', 'ATX': 'green', 'DNP': 'red'}  # Condition color mapping

        # Loop through each measure to process data and plot
        for measure in measures:
            combined_df_plotting = pd.DataFrame()
            file_paths = glob.glob(os.path.join(self.pupil_results, "Correlation_Pupil_EEG_*.csv"))
            
            # Process each file and prepare the combined DataFrame
            for file in file_paths:
                condition = os.path.basename(file).replace('Correlation_Pupil_EEG_', '').replace('.csv', '')
                df = pd.read_csv(file)
                df.set_index('EEG_channel_name', inplace=True)
                
                # Filter columns containing the measure and append condition to each column name
                selected_columns = df.filter(like=measure)
                selected_columns.rename(columns=lambda x: f"{x}_{condition}", inplace=True)
                
                if combined_df_plotting.empty:
                    combined_df_plotting = selected_columns
                else:
                    combined_df_plotting = pd.concat([combined_df_plotting, selected_columns], axis=1)

            # Now plot the data for this measure
            channels = combined_df_plotting.index.unique()
            fig = make_subplots(rows=len(channels), cols=1, subplot_titles=[f'Channel: {ch}' for ch in channels], shared_xaxes=False)

            for i, channel in enumerate(channels):
                for condition, color in colors.items():
                    condition_measure_columns = [col for col in combined_df_plotting.columns if measure in col and condition in col]
                    time_shifts = [int(col.split('_')[0]) for col in df.columns if col.split('_')[0].isdigit() and 'correlation' in col]
                    measure_data = combined_df_plotting.loc[channel, condition_measure_columns]

                    # Plot the data
                    fig.add_trace(
                        go.Scatter(
                            x=time_shifts,
                            y=measure_data,
                            mode='lines',
                            name=f"{condition}" if i == 0 else "",
                            showlegend=i == 0,  # Show legend only in the first subplot
                            line=dict(color=color),
                            hoverinfo='none',  # Disable default hover info
                            hovertemplate=f'{condition}<br>Time Shift: %{{x}} ms<br>Value: %{{y:.4f}}'
                        ),
                        row=i + 1, col=1)
                        
            # Update layout and axes
            fig.update_layout(
                height=300 * len(channels),
                width=800,
                title_text=f"Time Variation of {measure} Across Conditions",
                showlegend=True
            )
            fig.update_xaxes(title_text="Time Shift (ms)")
            fig.update_yaxes(title_text="Correlation Value" if measure == 'correlation' else "R-Squared Value", secondary_y=False)
            if measure != 'correlation':
                fig.update_yaxes(title_text="R-Squared Value", secondary_y=True)

            # Save the plot to HTML file
            output_html = os.path.join(self.pupil_results, f"{measure}_EEG_Plot_Interactive.html")
            fig.write_html(output_html)
    
    def plot_maximum_correlation_per_condition(self, mean_abs_value=0, max_key=None):
        """
        Plots the topographic map of the maximum correlation values for each condition.

        Parameters:
        mean_abs_value (float): The mean absolute value of correlations.
        max_key (str): The key for the maximum value.
        """
        # Load CSV files containing correlation data
        csv_files = glob.glob(os.path.join(self.pupil_results, 'Correlation_Pupil_EEG_*.csv'), recursive=False)

        # Methods to consider
        methods = ['correlation', 'first_order', 'second_order']

        # Initialize DataFrames for each condition
        PCB_correlations = pd.DataFrame()
        ATX_correlations = pd.DataFrame()
        DNP_correlations = pd.DataFrame()

        for file in csv_files:
            df = pd.read_csv(file)
            condition = file.split("_")[-1].replace(".csv", "")

            for method in methods:
                # Identify columns that contain the method in their names
                correlation_columns = [col for col in df.columns if method in col]

                # Calculate the mean of the absolute values for these columns
                mean_abs_values = {col: df[col].abs().mean() for col in correlation_columns}
                highest_abs_mean_column = max(mean_abs_values, key=mean_abs_values.get)
                highest_value_plot = int(highest_abs_mean_column.split("_")[0])

                # Prepare the path for correlation files
                value_in_s = highest_value_plot / 1000
                correlation_files = glob.glob(os.path.join(self.pupil_results, "**", f'*{condition}*{value_in_s}s*.csv'), recursive=True)

                for file in correlation_files:
                    current_df = pd.read_csv(file)
                    participant = os.path.dirname(file).split("/")[-1]
                    method_column = [col for col in current_df.columns if col.startswith(method)][0]

                    # Append the correlation data to the respective DataFrame
                    if condition == "PCB":
                        PCB_correlations["EEG_channel_names"] = current_df["EEG_channel_name"]
                        PCB_correlations[participant + "_" + method] = current_df[method_column]
                    elif condition == "DNP":
                        DNP_correlations["EEG_channel_names"] = current_df["EEG_channel_name"]
                        DNP_correlations[participant + "_" + method] = current_df[method_column]
                    elif condition == "ATX":
                        ATX_correlations["EEG_channel_names"] = current_df["EEG_channel_name"]
                        ATX_correlations[participant + "_" + method] = current_df[method_column]

                # Extract the correlation values for each channel
                correlation_values = df[highest_abs_mean_column].values
                channels = df['EEG_channel_name'].values

                # Create a montage for standard 10-20 system
                montage = mne.channels.make_standard_montage('biosemi64')

                # Create an Info object
                info = mne.create_info(channels.tolist(), sfreq=1000, ch_types='eeg')

                # Create an EvokedArray object
                evoked = mne.EvokedArray(correlation_values[:, np.newaxis], info)
                evoked.set_montage(montage)

                # Plot the topographic map without the draggable colorbar
                fig, ax = plt.subplots()
                im, _ = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=ax, show=False)

                # Add a colorbar manually
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(f'{method} Values')

                # Display the plot
                plt.title(f'Topographic map of {method}\nTime Delay: {highest_value_plot}ms')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                plot_path = os.path.join(self.pupil_results, f'topomap_{condition}_{method}_per_condition.pdf')
                fig.savefig(plot_path)
                plt.close()

    def plot_maximum_correlation_overall(self):
        """
        Plots the topographic map of the maximum correlation values for all conditions combined.
        """
        # Load CSV files containing correlation data
        csv_files = glob.glob(os.path.join(self.pupil_results, 'Correlation_Pupil_EEG_*.csv'), recursive=False)
        methods = ['correlation', 'first_order', 'second_order']

        PCB_correlations = pd.DataFrame()
        ATX_correlations = pd.DataFrame()
        DNP_correlations = pd.DataFrame()

        # Calculate the sum of absolute correlations for each method
        for method in methods:
            correlation_sums = {}
            value_counts = {}

            for file in csv_files:
                df = pd.read_csv(file)
                condition = file.split("_")[-1].replace(".csv", "")

                # Identify columns that contain the method name in their names and calculate the mean of the absolute values
                correlation_columns = [col for col in df.columns if method in col]

                for col in correlation_columns:
                    if col not in correlation_sums:
                        correlation_sums[col] = 0
                        value_counts[col] = 0

                    correlation_sums[col] += df[col].abs().sum()
                    value_counts[col] += df[col].notna().sum()

            # Calculate averages
            correlation_averages = {col: correlation_sums[col] / value_counts[col] for col in correlation_sums}

            # Find the time point with the highest average of absolute correlations
            max_key = max(correlation_averages, key=correlation_averages.get)
            max_value = correlation_averages[max_key]
            highest_value_plot = int(max_key.split("_")[0])
            value_in_s = highest_value_plot / 1000

            # Create a line plot
            plt.figure(figsize=(10, 6))
            plt.plot(list(correlation_averages.keys()), list(correlation_averages.values()), marker='o', linestyle='-', color='b')
            plt.axvline(x=max_key, color='r', linestyle='--', label=f'Highest: {max_key} ({max_value:.2f})')
            plt.xlabel('Time Points')
            plt.ylabel(f'{method} Average of Absolute Values')
            plt.title(f'Average of Absolute {method}')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.pupil_results, f'absolute_average_{method}_plot_over_time.pdf')
            plt.savefig(plot_path)

            # Load the files for the determined time point and create topographic maps
            correlation_files = glob.glob(os.path.join(self.pupil_results, "**", f'*{value_in_s}s*.csv'), recursive=True)

            self.create_plots(value_in_s, correlation_files, method)

            for file in correlation_files:
                current_df = pd.read_csv(file)

                if "PCB" in file:
                    condition = "PCB"
                elif "ATX" in file:
                    condition = "ATX"
                elif "DNP" in file:
                    condition = "DNP"

                participant = os.path.dirname(file).split("/")[-1]

                method_column = [col for col in current_df.columns if col.startswith(method)][0]

                if condition == "PCB":
                    PCB_correlations["EEG_channel_names"] = current_df["EEG_channel_name"]
                    PCB_correlations[participant + "_" + method] = current_df[method_column]
                elif condition == "DNP":
                    DNP_correlations["EEG_channel_names"] = current_df["EEG_channel_name"]
                    DNP_correlations[participant + "_" + method] = current_df[method_column]
                elif condition == "ATX":
                    ATX_correlations["EEG_channel_names"] = current_df["EEG_channel_name"]
                    ATX_correlations[participant + "_" + method] = current_df[method_column]

            ATX_average_df = self.calculate_averages(ATX_correlations, method)
            PCB_average_df = self.calculate_averages(PCB_correlations, method)
            DNP_average_df = self.calculate_averages(DNP_correlations, method)

            # Not the best solution but does the job nicely
            ATX_average_df["Condition"] = "ATX"
            PCB_average_df["Condition"] = "PCB"
            DNP_average_df["Condition"] = "DNP"

            list_of_df = [ATX_average_df, PCB_average_df, DNP_average_df]

            for df in list_of_df:
                # Extract the correlation values for each channel
                correlation_values = df['Average'].values
                channels = df['EEG_channel_names'].values

                condition = df['Condition'].values[0]

                # Create a montage for standard 10-20 system
                montage = mne.channels.make_standard_montage('biosemi64')

                # Create an Info object
                info = mne.create_info(channels.tolist(), sfreq=1000, ch_types='eeg')

                # Create an EvokedArray object
                evoked = mne.EvokedArray(correlation_values[:, np.newaxis], info)
                evoked.set_montage(montage)

                # Plot the topographic map without the draggable colorbar
                fig, ax = plt.subplots()
                im, _ = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=ax, show=False)

                # Add a colorbar manually
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(f'{method} Values')

                # Display the plot
                plt.title(f'Topographic map of {method}\nCondition: {condition}\nTime Delay: {highest_value_plot}ms')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                plot_path = os.path.join(self.pupil_results, f'topomap_{method}_{condition}_max_sum.pdf')
                fig.savefig(plot_path)
                plt.close(fig)

            electrode_dfs = {}

            # Create lists to hold p-values and masks for significant electrodes
            p_values_ATX = []
            p_values_DNP = []

            bayes_factors_ATX = []
            bayes_factors_DNP = []

            masks_ATX = []
            masks_DNP = []

            # Iterate over each electrode
            for electrode in PCB_correlations['EEG_channel_names']:
                # Filter columns that contain the method name
                method_columns = [col for col in PCB_correlations.columns if method in col]

                # Create a new DataFrame for the electrode
                electrode_df = pd.DataFrame()

                rows = []

                # Iterate over method columns and add data to the new DataFrame
                for col in method_columns:
                    participant = col.split('_')[0]

                    value_PCB = PCB_correlations.loc[PCB_correlations['EEG_channel_names'] == electrode, col].values[0]
                    value_ATX = ATX_correlations.loc[ATX_correlations['EEG_channel_names'] == electrode, col].values[0]
                    value_DNP = DNP_correlations.loc[PCB_correlations['EEG_channel_names'] == electrode, col].values[0]
                    rows.append({"Participant": participant, "PCB_Value": value_PCB,
                                "ATX_Value": value_ATX, "DNP_Value": value_DNP})

                electrode_df = pd.concat([electrode_df, pd.DataFrame(rows)], ignore_index=True)

                # Perform paired t-tests for PCB vs ATX and PCB vs DNP
                stats_test_ATX = pg.ttest(electrode_df['PCB_Value'], electrode_df['ATX_Value'], paired=True)
                stats_test_DNP = pg.ttest(electrode_df['PCB_Value'], electrode_df['DNP_Value'], paired=True)

                p_val_ATX = stats_test_ATX['p-val'].values[0]
                p_val_DNP = stats_test_DNP['p-val'].values[0]

                # Append p-values to lists
                p_values_ATX.append(p_val_ATX)
                p_values_DNP.append(p_val_DNP)

                # Extract Bayes Factors
                bf_ATX = float(stats_test_ATX['BF10'].values[0])
                bf_DNP = float(stats_test_DNP['BF10'].values[0])

                # Append Bayes Factors to lists
                bayes_factors_ATX.append(bf_ATX)
                bayes_factors_DNP.append(bf_DNP)

            # Convert p-value lists to numpy arrays
            p_values_ATX = np.array(p_values_ATX)
            p_values_DNP = np.array(p_values_DNP)

            # Multiple comparisons correction using FDR (Benjamini-Hochberg)
            _, p_values_ATX, _, _ = multipletests(p_values_ATX, alpha=0.05, method='fdr_bh')
            _, p_values_DNP, _, _ = multipletests(p_values_DNP, alpha=0.05, method='fdr_bh')

            bayes_factors_ATX = np.array(bayes_factors_ATX)
            bayes_factors_DNP = np.array(bayes_factors_DNP)

            # Create masks for significant electrodes (e.g., p-value < 0.05)
            mask_ATX = p_values_ATX < 0.05
            mask_DNP = p_values_DNP < 0.05

            mask_ATX_bayesian = bayes_factors_ATX > 3
            mask_DNP_bayesian = bayes_factors_DNP > 3

            if any(mask_ATX == True) or any(mask_DNP == True):
                print(method)
                print("There is at least one significant difference between PCB and Experimental")

            if any(mask_ATX_bayesian == True) or any(mask_DNP_bayesian == True):
                print(method)
                print("There is at least one significant difference bayesian between PCB and Experimental")

            # As there was no significant mask for the correlation, I did not continue creating a mask.

    "Below are some support functions that help make the code simpler"
    
    def calculate_averages(self, dataframe, method):
        """
        Calculates the average values for the specified method across all columns
        containing the method name in the given dataframe.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing correlation values.
        method (str): The method name to filter columns.

        Returns:
        pd.DataFrame: A dataframe containing EEG channel names and their average values.
        """
        # Ensure EEG_channel_names only contains valid strings
        valid_dataframe = dataframe[dataframe["EEG_channel_names"].apply(lambda x: isinstance(x, str))]

        # Extract the channel names
        channels = valid_dataframe["EEG_channel_names"]

        # Filter columns that contain the specified method name
        columns_to_average = [col for col in dataframe.columns if method in col]

        # Select only the filtered columns and drop the EEG channel names column
        data_to_average = valid_dataframe[columns_to_average]

        # Calculate the mean across the selected columns
        averages = data_to_average.mean(axis=1)

        # Create a new DataFrame for the averages with channels
        averages_df = pd.DataFrame({
            "EEG_channel_names": channels,
            "Average": averages
        })

        return averages_df

    def create_plots(self, max_time_point, correlation_files, method):
        """
        Creates topographic plots of EEG correlation values for multiple participants and conditions.

        Parameters:
        max_time_point (float): The time point (in seconds) for which to create the plots.
        correlation_files (list of str): List of file paths containing correlation data.
        method (str): The method name to filter columns.
        """
        # Extract unique participants from the correlation files
        participants = sorted(set([os.path.basename(file).split('_')[1] for file in correlation_files]))

        # Create subplots for each participant and condition
        fig, axes = plt.subplots(nrows=len(participants), ncols=3, figsize=(15, 5 * len(participants)))

        for i, participant in enumerate(participants):
            for j, condition in enumerate(['PCB', 'ATX', 'DNP']):
                # Pattern to match the correlation files for the current participant and condition
                file_pattern = f"{participant}_{condition}_shift_{max_time_point}s_df.csv"

                # Find matching files
                matching_files = [file for file in correlation_files if file_pattern in file]

                if not matching_files:
                    continue

                df = pd.read_csv(matching_files[0])
                method_column = [col for col in df.columns if col.startswith(method)][0]

                ax = axes[i, j]

                # Create an Info object with the correct montage
                montage = mne.channels.make_standard_montage('biosemi64')
                info = mne.create_info(df['EEG_channel_name'].tolist(), sfreq=1000, ch_types='eeg')
                info.set_montage(montage)

                # Plot topographic map
                im, _ = mne.viz.plot_topomap(df[method_column].values, info, axes=ax, show=False)
                ax.set_title(f"{participant} - {condition}")

                # Add a colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f'{method} Values')

        plt.tight_layout()
        plot_path = os.path.join(self.pupil_results, f'multiple_people_{method}_max_correlation_{max_time_point}.pdf')
        plt.savefig(plot_path, dpi=300)
        plt.close()









        


        