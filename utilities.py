from __future__ import division

# import python moduls
from IPython import embed as shell
import numpy as np
import copy
import mne
import pickle
import time as t
import pandas as pd

class Utilities(object):
    """
    This class contains all custom written general functionalities.
    """

    def trial_selection(self,hdf5_filename,block_trial_indices, alias, no_std_cutoff=3,
                degree_cutoff=1.5, analyze_eye='L',required_phase=3,bi = 0,amplitudes=[],
                n_directions = 6,minimum_saccade_size=3):
        """
        select_trials returns a boolean array with 1's for trials to use. Trials are deselected when:
        1. there was a blink in trialphase 3 of that trial (the phase where the script polls for a saccade)
        2. the amplitude of the saccade was less then no_std_cutoff median absolute deviations
        3. the starting gaze position is more than 'degree_cutoff' distance off
        """

        # import required datafiles
        with pd.get_store(hdf5_filename) as h5_file:
            saccade_table = h5_file['%s/saccades_per_trial' % alias]
            trial_phases = h5_file['%s/trial_phases' % alias]
            blink_table = h5_file['%s/blinks_from_message_file' % alias]
            params = h5_file['%s/parameters' % alias]
            pixels_per_degree = params['pixels_per_degree'][0]

        saccade_table = saccade_table[saccade_table.eye == analyze_eye].set_index('trial')
        trials_this_block = block_trial_indices[bi]

        # initialize outcome array
        n_trials = len(trials_this_block)
        trials2use = np.ones(n_trials)

        # 1. deselect trials that occured during a blink
        start_times = np.array(trial_phases.trial_phase_EL_timestamp[trial_phases.trial_phase_index ==required_phase])[trials_this_block]
        end_times = np.array(trial_phases.trial_phase_EL_timestamp[trial_phases.trial_phase_index == required_phase+1])[trials_this_block]

        for blink_start in blink_table.start_timestamp:
            blink_trial_index = np.where((blink_start > start_times) * (blink_start < end_times))[0]
            if blink_trial_index != []:
                trials2use[blink_trial_index[0]] = 0

        # 2. deselect trials when amplitude is more than no_std_cutoff away from median in this block:
        this_block_data = amplitudes
        amp_outliers = (self.detect_inliers_mad(this_block_data,outlier_num_stds=no_std_cutoff)==False)
        trials2use[np.array(amp_outliers+(this_block_data<minimum_saccade_size)).astype(bool)] = 0

        # 3. deselect trials when gaze offset is more than degree_cutoff away from median start position:
        saccade_startpoints = np.array([sv for sv in saccade_table['expanded_start_point'][trials_this_block]])
        median_saccade_startpoints = np.median(np.reshape(saccade_startpoints,(-1,n_directions,2)),axis=0)

        gaze_offset = np.linalg.norm(np.array([saccade_startpoints[si] - median_saccade_startpoints[np.mod(si,n_directions)] for si in range(len(saccade_startpoints))])/pixels_per_degree,axis=1)
        gaze_outliers = gaze_offset>degree_cutoff

        trials2use[gaze_outliers] = 0


        return trials2use.astype(bool)

    def detect_inliers_mad(self,data,outlier_num_stds=3):
        
        """
        Detects inliers based on the Median Absolute Deviation, see:
        http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
        https://en.wikipedia.org/wiki/Median_absolute_deviation

        It calculates the MAD for left and right of median separately, 
        ensuring that it can handle skewed data.
    
        :param data: input data
        :type data: 1-D array
        :param outlier_num_stds: number of standard deviations to include
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the outlier rejections
        :type generate_diagnostics: bool

        :return inliers: returns boolean array of inliers
        :return type: same shape as data
        """

        # first calculate the distances for the median for left and right of median separately:
        mad_left = np.median(np.abs(data[data<np.median(data)]-np.median(data)))
        mad_right = np.median(np.abs(data[data>np.median(data)]-np.median(data)))

        # constant assuming normality
        k = 1.4826 

        # then determine thresholds 
        left_threshold = k*mad_left*outlier_num_stds
        right_threshold = k*mad_right*outlier_num_stds

        # find which values fall below the thresholds
        left_inliers = (np.abs(data[data<np.median(data)]-np.median(data))<left_threshold)
        right_inliers = (np.abs(data[data>np.median(data)]-np.median(data))<right_threshold)

        # and put them together
        inliers = np.ones_like(data).astype(bool)
        inliers[data<np.median(data)] = left_inliers
        inliers[data>np.median(data)] = right_inliers

        return inliers

    def p_val_from_bootstrap_dist(self,distribution,test_value=0,two_tailed=True):
        """
        Finds p-value for hypothesis that the distribution is not different 
        from the test_value.

        :param distribution: distribution of bootstrapped parameter
        :type distribution: 1-D array
        :param test_value: value to test distribution against
        :type test_value: float
        :param two_tailed: if True, returns two-tailed test, else one-tailed
        :type two_tailed: bool

        :return p-val: p-val
        :type p-val: float
        """

        # see which part of the distribution falls below / above test value:
        proportion_smaller_than_test_value = np.sum(np.array(distribution) < test_value) / len(distribution)
        proportion_larger_than_test_value = np.sum(np.array(distribution) > test_value) / len(distribution)

        # take minimum value as p-val:
        p = np.min([proportion_smaller_than_test_value,proportion_larger_than_test_value])
        
        # this yields a one-tailed test, so multiply by 2 if we want a two-tailed p-val:
        if two_tailed:
            p*=2

        return p
