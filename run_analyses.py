from __future__ import division

import os, sys, datetime, pickle
import subprocess, time
import pp

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import pandas as pd

from group_level import HexagonalSaccadeAdaptationGroupLevelAnalyses

from IPython import embed as shell

##############################################
# this directory should point to the downloaded data
data_dir = '/home/shared/2017/eye_movements/group_level/all_data'
output_dir = '/home/shared/2017/eye_movements/group_level/new_figs'
##############################################

# experimental parameters
conditions = {'react_updown':1,'react_downup':2,'scanning_updown':3,'scanning_downup':4}
saccade_amp =10
up_gain = 1.33
down_gain = 0.75
screen_res = [1024,786]
trials_per_block = [48,96,96,48]
n_directions = 6
trials_passed = np.hstack([0,np.cumsum(trials_per_block)])
block_trial_indices = [np.arange(trials_per_block[ti]) + trials_passed[ti] for ti in range(len(trials_per_block))]

# create subject array
n_subs = 12
subjects = [type('', (), {})() for i in range(n_subs)]
for i,s in enumerate(subjects):
	s.initials = str(i+1)

def group_level_analyses(data_folder,conditions):
	
	aliases = [c for c in conditions.keys()]
	hsas_gl = HexagonalSaccadeAdaptationGroupLevelAnalyses(subjects =subjects, aliases=aliases,data_folder = data_folder, output_dir=output_dir,block_trial_indices=block_trial_indices,saccade_amp=saccade_amp,n_directions=n_directions,up_gain=up_gain,down_gain=down_gain,screen_res=screen_res)

	# Figure 2:
	hsas_gl.plot_saccade_latencies()

	# Figure 3:
	hsas_gl.plot_adaptation()

	# Figure 4:
	hsas_gl.fit_timescale_to_adaptation(reps=int(1e5))# this takes a while - reduce reps if not enough computing time is available
	hsas_gl.plot_block_jumps()

	#Figure 5:
	hsas_gl.plot_fit_results(fit_learning_parameters=False)
	hsas_gl.plot_fit_results(fit_learning_parameters=True)
	
	#Figure 6:
	hsas_gl.compare_fit_gains(fit_learning_parameters=False)
	hsas_gl.compare_fit_gains(fit_learning_parameters=True)

	# Figure 7:
	hsas_gl.plot_seen_reports()


def main():
	group_level_analyses(data_dir,conditions)

if __name__ == '__main__':
	main()

# 