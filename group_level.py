#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Tomas Knapen on 2011-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
from __future__ import division
import os, math
import pickle
import numpy as np
import scipy as sp
import scipy.stats as stats
# import python moduls
import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import seaborn as sns
sns.set(style="white")
from decimal import Decimal

import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
from scipy.optimize import curve_fit
from scipy import stats, polyval, polyfit
from lmfit import minimize, Parameters, Parameter, report_fit
import copy
from joblib import Parallel, delayed
import itertools
from itertools import chain
import statsmodels.stats.api as sms
import statsmodels.api as sm

# import logging, logging.handlers, logging.config
import sys
from hedfpy import EDFOperator, HDFEyeOperator, EyeSignalOperator
from hedfpy.EyeSignalOperator import detect_saccade_from_data

from IPython import embed as shell

from utilities import Utilities
mpl.rc_file_defaults()

# for plos one:
mpl.rcParams.update({'font.size': 8})
mpl.rcParams.update({'axes.labelsize': 8})
mpl.rcParams.update({'figure.titlesize': 8})
mpl.rcParams.update({'axes.titlesize': 8})

def smith_model(params,data,adaptation_goal,fit=True):

    # params 
    fast_forget = params['fast_forget'].value
    slow_forget = params['fast_forget'].value + params['delta_forget'].value

    slow_learn = params['slow_learn'].value
    fast_learn = params['slow_learn'].value+params['delta_learn'].value

    # initialize states
    adap = {}
    adap['slow'] = np.zeros(len(data)+1)
    adap['fast'] = np.zeros(len(data)+1)
    adap['both'] = np.zeros(len(data)+1)

    # now loop over errors and compute state
    for i in range(len(data)):
        error = adaptation_goal-(adap['fast'][i]+adap['slow'][i]) 
        
        adap['fast'][i+1] = (fast_forget * adap['fast'][i] + fast_learn * error)
        adap['slow'][i+1] = (slow_forget * adap['slow'][i] + slow_learn * error)

    # multiply with gain and delete last value 
    adap['fast'] = adap['fast'][:-1] * params['fast_gain'].value
    adap['slow'] = adap['slow'][:-1] * params['slow_gain'].value
    # put states together
    adap['both'] = adap['slow']+adap['fast']

    if fit:
        # check if all parameters are within range:
        k=0
        for value in [fast_learn,fast_forget,slow_learn,slow_forget]:
            if (value < 0) + (value > 1):
                k+=1

        # if not, return huge residuals
        if k > 0:
            residuals = np.ones_like(data)*10000
        # if they are, return regular residuals
        else:
            valid_trials = ~np.isnan(data)        
            residuals = data[valid_trials] - adap['both'][valid_trials]

        return residuals
    else:
        return adap

class HexagonalSaccadeAdaptationGroupLevelAnalyses(object):
    """
    Instances of this class can be used to execute group level analyses for Hexagonal Saccade Adaptation experiments
    """
    def __init__(self, subjects, aliases,data_folder,output_dir,block_trial_indices=[0],saccade_amp=10,n_directions=6,all_block_colors=['r'],up_gain=1.33,down_gain=0.75,screen_res=[1024,768] ):
        
        self.subjects = subjects
        self.data_dir = data_folder
        self.plot_dir = output_dir
        self.screen_res = screen_res

        self.up_gain = up_gain
        self.down_gain = down_gain
        self.nr_blocks = len(block_trial_indices)
        self.saccade_amplitude = saccade_amp
        self.n_directions = n_directions
        self.block_trial_indices = block_trial_indices
        self.n_trials_per_block = [len(self.block_trial_indices[b]) for b in range(self.nr_blocks)]
        self.aliases=aliases

        try:
            os.mkdir(self.plot_dir)
        except OSError:
            print('failed to create dirs')
            pass

        # helper object:
        self.CU = Utilities()

    def get_data(self,analyze_eye='L'):
        
        self.all_data = {}
        for ai, alias in enumerate(self.aliases):
            print()
            self.all_data[alias] = {}
            for s in self.subjects:

                sys.stdout.write('reading data for %s subject %s/%d\r'%(alias,s.initials,len(self.subjects)))
                sys.stdout.flush()
                # print 'getting data for %s subject %s/%d'%(alias,s.initials,len(self.subjects))

                self.all_data[alias][s.initials] = {}
                this_s_fn = os.path.join(self.data_dir, s.initials + '.hdf5')
                with pd.get_store(this_s_fn) as h5_file:
                    saccade_table = h5_file['%s/saccades_per_trial'%alias]
                    parameters = h5_file['%s/parameters' % alias]
                    pixels_per_degree = parameters['pixels_per_degree'][0]
                    trial_phases = h5_file['%s/trial_phases' % alias]

                for i in range(self.nr_blocks):
                    self.all_data[alias][s.initials][i] = {}

                    this_block_n_trials = len(self.block_trial_indices[i])
                    trials_passed = sum([len(b) for b in self.block_trial_indices][:i])

                    # get saccade start and end point
                    saccade_startpoints = np.array([sv for sv in saccade_table['expanded_start_point'][(saccade_table['block'] == i)]])
                    saccade_endpoints = np.array([sv for sv in saccade_table['expanded_end_point'][(saccade_table['block'] == i)]])
                   
                    # estimate the dot position from median saccade start positions
                    median_start_points = np.median(np.reshape(saccade_startpoints,(-1,self.n_directions,2)),axis=0)
                    # rotate so first start position matches first end position
                    median_start_points_rotated = np.vstack([median_start_points[1:],median_start_points[0]])
                    # then compute error
                    errors = np.array([saccade_endpoints[si] - median_start_points_rotated[np.mod(si,self.n_directions)] for si in range(len(saccade_endpoints))])
                    # now project the error vectors onto unit vectors in the direction of the intended saccade
                    angles = np.radians(np.array([135,90,45,315,270,225]))
                    unit_vectors = [[np.cos(angle),np.sin(angle)] for angle in angles]
                    errors_from_post_target = np.array([np.dot(errors[vi],unit_vectors[np.mod(vi,self.n_directions)]) for vi in range(len(errors))]) / pixels_per_degree 

                    # finally subtract from that, the distance from the pre to the post-saccadic target
                    errors_from_pre_target = errors_from_post_target - (self.saccade_amplitude-self.saccade_amplitude*parameters['adaptation_gain'][trials_passed+1])
                    # and add saccade amplitude so it's errors from 10 (not from 0)
                    this_block_data = self.saccade_amplitude + errors_from_pre_target

                    # define valid trials
                    valid_trials = self.CU.trial_selection(this_s_fn,self.block_trial_indices, alias,bi=i,amplitudes=this_block_data)
                    this_block_data[~valid_trials] = np.nan

                    # define baseline
                    if i == 0:
                        n_trials = len(this_block_data)/self.n_directions
                        baseline_dirs = np.nanmedian(np.reshape(this_block_data,(-1,self.n_directions)),axis=0)
                    reps = len(this_block_data)/self.n_directions
                    baseline = np.tile(baseline_dirs,reps)
                    baseline[np.isnan(baseline)]=0

                    # now apply the baseline correction
                    this_block_data /= baseline          

                    # compute moving average
                    ma_steps = self.n_directions
                    pad_ma= False
                    if pad_ma:
                        data_to_fit = np.hstack([np.ones(ma_steps)*np.nan,this_block_data,np.ones(ma_steps)*np.nan])
                    else:
                        data_to_fit = this_block_data
                    ma_y = np.array([np.nanmean(data_to_fit[ti:ti+ma_steps]) for ti in range(len(data_to_fit)-ma_steps+1)])
                    ma_x = np.linspace(0,len(this_block_data),len(ma_y))                       

                    # save data to dict
                    self.all_data[alias][s.initials][i]['data'] = this_block_data
                    self.all_data[alias][s.initials][i]['ma_x'] = ma_x
                    self.all_data[alias][s.initials][i]['ma_y'] = ma_y

                    # also retrieve seen reports
                    seen = (parameters['seen'][self.block_trial_indices[i]]+1)/2
                    seen[~valid_trials] = np.nan
                    self.all_data[alias][s.initials][i]['seen'] = seen

                    # and add saccade onset latency
                    # phase 3 was start of saccade polling, trial phase 4 was detection of saccade and stepping target
                    start_times = np.array(trial_phases.trial_phase_EL_timestamp[trial_phases.trial_phase_index == 3])[self.block_trial_indices[i]]
                    end_times = np.array(trial_phases.trial_phase_EL_timestamp[trial_phases.trial_phase_index == 4])[self.block_trial_indices[i]]
                    saccade_latencies = end_times-start_times         
                    self.all_data[alias][s.initials][i]['latencies'] = saccade_latencies[valid_trials]

        return self.all_data

    def plot_saccade_latencies(self):

        if not hasattr(self,'all_data'):
            self.get_data()

        figdir = os.path.join(self.plot_dir,'Fig2')
        if not os.path.isdir(figdir): os.mkdir(figdir)

        ##############################################################################
        # get the latencies together in one dict (and df for aov)
        ##############################################################################
        # block_idx = 1
        # first get gains

        latency_df = {key: [] for key in ['subject','condition','direction','latency']}
        for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]):
            for alias in self.aliases:
                these_latencies = np.hstack([self.all_data[alias][subject][bi]['latencies'] for bi in range(self.nr_blocks)])
                # these_latencies = self.all_data[alias][subject][block_idx]['latencies']
                latency_df['subject'].append(subi)
                latency_df['condition'].append(alias.split('_')[0])
                if 'updown' in alias:
                    latency_df['direction'].append('up')
                elif 'downup' in alias:
                    latency_df['direction'].append('down')
                latency_df['latency'].append(np.nanmean(these_latencies))

        ##############################################################################
        # perform the anova
        ##############################################################################

        # # let's put it in a df
        import pyvttbl as pt
        df = pt.DataFrame(latency_df)
        aov = df.anova('latency', sub='subject', wfactors=['condition','direction'])
        print(aov)

        ##############################################################################
        # plot an overview of the different latencies in all conditions
        ##############################################################################

        x_offset = .15
        # now plot
        f = pl.figure(figsize=(2.6,2))
        s = f.add_subplot(1,1,1)
        for ai, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):

            # get up gains
            react_latencies = [np.nanmean(np.hstack([self.all_data[alias_combo[0]][sub.initials][bi]['latencies'] for bi in range(self.nr_blocks)])) for sub in self.subjects]
            react_mean = np.mean(react_latencies)
            react_ci=sms.DescrStatsW(react_latencies).tconfint_mean()

            # get fast errors
            scanning_latencies = [np.nanmean(np.hstack([self.all_data[alias_combo[1]][sub.initials][bi]['latencies'] for bi in range(self.nr_blocks)])) for sub in self.subjects]
            scanning_mean = np.mean(scanning_latencies)
            scanning_ci=sms.DescrStatsW(scanning_latencies).tconfint_mean()

            j = ai*x_offset
            pl.plot(np.linspace(j-x_offset/4,j+x_offset/4,len(react_latencies)),react_latencies,'o',mec='w',ms=4,color=['g','r'][ai],alpha=0.5)
            pl.plot(np.linspace(1+j-x_offset/4,1+j+x_offset/4,len(scanning_latencies)),scanning_latencies,'o',mec='w',ms=4,color=['g','r'][ai],alpha=0.5)

            # plot
            pl.plot([0+j,1+j],[react_mean,scanning_mean],marker=['o','s'][ai],color=['g','r'][ai],mec='w',label=['down','up'][ai],lw=1.5,ms=8)
          
            pl.plot([0+j,0+j],react_ci,color='w',lw=2)
            pl.plot([0+j,0+j],react_ci,color=['g','r'][ai],lw=1.5)

            pl.plot([1+j,1+j],scanning_ci,color='w',lw=2)
            pl.plot([1+j,1+j],scanning_ci,color=['g','r'][ai],lw=1.5)

            if ai == 1:
                pl.legend(loc='upper left')

            sns.despine(offset=2)
            pl.xticks([0+x_offset/2,1+x_offset/2],['react','scanning'])
            pl.yticks([0,100,200,300,400,500])
            # pl.ylim(0,4)
            # pl.yticks([0,4])
            pl.ylabel('saccade latency (ms)')
            pl.xlim(-0.5,1.65)

        f.tight_layout()
        f.savefig(os.path.join(figdir,'latencies.pdf'))
        pl.close()

    def plot_adaptation(self):
        
        if not hasattr(self, 'all_data'):
            self.get_data()

        figdir = os.path.join(self.plot_dir,'Fig3')
        if not os.path.isdir(figdir): os.mkdir(figdir)

        # for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]+['avg']):
        for subi,subject in enumerate(['avg']):#[self.subjects[si].initials for si in range(len(self.subjects))]+['avg']):
            for aii, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
                f = pl.figure(figsize=(2.75,2))
                for ai, alias in enumerate(alias_combo):
                    s = f.add_subplot(2,1,ai+1)
                    pl.axhline(1,color='k',ls='--',lw=0.5)

                    if 'downup' in alias:
                        colors = ['k','g','r','k']
                    else:
                        colors = ['k','r','g','k']

                    for bi in range(self.nr_blocks):

                        tp = np.hstack([0,np.cumsum(self.n_trials_per_block)])[bi]
                        pl.axvline(tp,color='k',linewidth=0.5)                        

                        if subject == 'avg':
                            # this_block_data = np.nanmean([self.all_data[alias][sub.initials][bi]['data']  for sub in self.subjects],axis=0)
                            x_data = np.array([self.all_data[alias][sub.initials][bi]['ma_x'] for sub in self.subjects])
                            y_data = np.array([self.all_data[alias][sub.initials][bi]['ma_y'] for sub in self.subjects])
                            this_block_x = np.nanmean(x_data,axis=0)
                            this_block_y = np.nanmean(y_data,axis=0)
                            this_block_se = np.nanstd(y_data,axis=0)/np.sqrt(len(self.subjects)-1)*stats.t.ppf(1-0.025, len(self.subjects)-1)

                            # pl.plot(tp+np.arange(len(this_block_data)),this_block_data,'o',color=colors[bi],alpha=.1,ms=3,mec='w')
                            pl.fill_between(tp+this_block_x,np.ravel([this_block_y+this_block_se]),np.ravel([this_block_y-this_block_se]),color=colors[bi],alpha=.2)
                            pl.plot(tp+this_block_x,this_block_y,lw=1,c=colors[bi],mec=colors[bi])

                            ylims = [.75,1.15]

                        else:
                            this_block_data = (self.all_data[alias][subject][bi]['data'] )
                            ma_x = self.all_data[alias][subject][bi]['ma_x']
                            ma_y = self.all_data[alias][subject][bi]['ma_y']

                            pl.plot(tp+np.arange(len(this_block_data)),this_block_data,'o',color=colors[bi],alpha=.2,ms=2,mec='w')
                            pl.plot(tp+ma_x,ma_y,lw=1,c=colors[bi],alpha=.1,mec=colors[bi])

                            ylims = [.5,1.5]

                        pl.ylim(ylims)

                        if ai ==1:
                            pl.xticks(np.hstack([0,np.cumsum(self.n_trials_per_block)]))
                        else:
                            pl.xticks([])
                        
                        # if bi == 0:
                        pl.yticks([ylims[0],1,ylims[1]])                            
                        # else:
                            # pl.yticks([])

                        if (ai==1):
                            pl.ylabel('saccade gain')
                            pl.xlabel('trial #')

                        sns.despine(offset=2)

                pl.tight_layout()
                pl.savefig(os.path.join(figdir,'%s_adaptation_sub_%s.pdf'%(alias_combo[0].split('_')[-1],subject)))
                pl.close()

    def fit_timescale_to_adaptation(self,reps=int(1e4),block_idx = 1):

        if not hasattr(self, 'all_data'):
            self.get_data()

        figdir = os.path.join(self.plot_dir,'Fig4')
        if not os.path.isdir(figdir): os.mkdir(figdir)

        # for linear interpolation:
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        # fit function
        def model(params,data,x):

            # intercept = params['offset'].value-1
            # fit = params['offset'].value+params['intercept'].value*np.exp(-x/params['slope'].value)
            fit = (1-params['gain'].value) * (np.exp(-x*params['timescale'].value)-1) + 1

            params['endpoint'].value = fit[-1]
            params['startpoint'].value = fit[0]

            residuals = data - fit

            return residuals
        
        permutations = np.random.randint(0,len(self.subjects),(reps,len(self.subjects)))
        sub_ids = np.array([self.subjects[si].initials for si in range(len(self.subjects))])
        timescales = {};startpoints = {};endpoints = {};predictions={};datadf={}
        for ai,alias in enumerate(self.aliases):
            print()
            timescales[alias] = []; startpoints[alias] = []; endpoints[alias] = []; predictions[alias] = []; datadf[alias] =[]
            for pi,perm in enumerate(np.vstack([permutations,np.arange(len(self.subjects))])):

                if np.mod(pi,int(reps/10))==0:
                    sys.stdout.write('done %d/%d reps\r'%(pi,reps))
                    sys.stdout.flush()
                these_subs = sub_ids[perm]
                
                data = np.nanmean([np.array(self.all_data[alias][sub][block_idx]['data']) for sub in these_subs],axis=0)

                x = np.arange(len(data))

                # linearly interpolate nans if present:
                nans, xnans = nan_helper(data)
                data[nans]= np.interp(xnans(nans), xnans(~nans), data[~nans])

                # setup fit parameters
                params = Parameters()
                params.add('timescale', value=0.05,min=0,max=1)
                if 'downup' in alias:
                    params.add('gain', value=0.25)
                elif 'updown' in alias:
                    params.add('gain', value=1.1)

                # these are just for prescription
                params.add('startpoint', value=0)
                params.add('endpoint', value=0)

                # perform fit
                res=minimize(model, params, kws={'data':data,'x':x})

                prediction = data-model(res.params,data,x)

                timescales[alias].append(res.params['timescale'].value)
                startpoints[alias].append(res.params['startpoint'].value) 
                endpoints[alias].append(res.params['endpoint'].value) 
                predictions[alias].append(prediction) 
                datadf[alias].append(data) 

        f = pl.figure(figsize=(1.2,2.75))
        for ai, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
            s = f.add_subplot(2,1,ai+1)

            for alias in alias_combo:
                x_data = np.array([self.all_data[alias][sub.initials][block_idx]['ma_x'] for sub in self.subjects])
                y_data = np.array([self.all_data[alias][sub.initials][block_idx]['ma_y'] for sub in self.subjects])
                this_block_x = np.nanmean(x_data,axis=0)
                this_block_y = np.nanmean(y_data,axis=0)  
                this_block_se = np.nanstd(y_data,axis=0)/np.sqrt(len(self.subjects)-1)*stats.t.ppf(1-0.025, len(self.subjects)-1)

                color = ['orange','c'][np.where(alias.split('_')[0]==np.array(['react','scanning']))[0][0]]
                label = ['reactive','scanning'][np.where(alias.split('_')[0]==np.array(['react','scanning']))[0][0]]
                ls = ['--','-'][np.where(alias.split('_')[0]==np.array(['react','scanning']))[0][0]]
                pl.fill_between(this_block_x,np.ravel([this_block_y+this_block_se]),np.ravel([this_block_y-this_block_se]),color=color,alpha=.2)
                pl.plot(x,np.array(predictions[alias])[-1],lw=1.5,c=color,ls=ls,label=label)
                
            if 'downup' in alias:
                ylims = [.75,1]
            else:
                ylims = [1,1.15]
            pl.ylim(ylims)

            # when to draw x axis
            pl.xticks([0,self.n_trials_per_block[block_idx]])
            pl.xlim([0,self.n_trials_per_block[block_idx]])
            sns.despine(offset=2)

            # when to draw y axis
            pl.yticks([ylims[0],1,ylims[1]])
            pl.ylabel('saccade gain')
            if ai == 1:
                pl.xlabel('trial #')
            else:
                pl.xticks([])

        pl.tight_layout()
        pl.savefig(os.path.join(figdir,'data_predictions.pdf'))
        pl.close()

        f = pl.figure(figsize=(1.2,2.75))
        for ai, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
            s = f.add_subplot(2,1,ai+1)
      
            for ac,alias in enumerate(alias_combo):          

                label = ['reactive','scanning'][np.where(alias.split('_')[0]==np.array(['react','scanning']))[0][0]]                
                sns.kdeplot(np.array(timescales[alias])[:-1],shade=True,color=['orange','c'][ac],linestyle=['--','-'][ac],lw=1.5)#,label=label,lw=1.5)
            

            pl.ylabel('density (a.u.)')
            if ai == 1:
                pl.xlabel('timescale')
                pl.xticks([0,0.5])      
            else:
                pl.xticks([])            

            pl.yticks([0,s.get_ylim()[1]])

            pl.xlim(0,0.5)
            sns.despine(offset=2)
        pl.tight_layout()
        pl.savefig(os.path.join(figdir,'timescales.pdf'))
        pl.close()    
        
        # test condition differences
        permutations = np.random.randint(0,len(self.subjects),(reps,len(self.subjects)))
        sub_ids = np.array([self.subjects[si].initials for si in range(len(self.subjects))])
        timescale_diff = {};startpoint_diff = {};endpoint_diff={}
        for ai, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
            print()
            if 'downup' in alias_combo[0]:
                direction = 'down'
            else:
                direction = 'up'
            timescale_diff[direction] = []; startpoint_diff[direction] = []; endpoint_diff[direction] = []
            
            for pi,perm in enumerate(permutations):

                if np.mod(pi,int(reps/10))==0:
                    sys.stdout.write('done %d/%d reps\r'%(pi,reps))
                    sys.stdout.flush()
                these_subs = sub_ids[perm]
                
                # first react data
                data_react = np.nanmean([np.array(self.all_data[alias_combo[0]][sub][block_idx]['data']) for sub in these_subs],axis=0)
                nans, xnans = nan_helper(data_react)
                data_react[nans]= np.interp(xnans(nans), xnans(~nans), data_react[~nans])
                
                # first scanning data
                data_scanning = np.nanmean([np.array(self.all_data[alias_combo[1]][sub][block_idx]['data']) for sub in these_subs],axis=0)
                nans, xnans = nan_helper(data_scanning)
                data_scanning[nans]= np.interp(xnans(nans), xnans(~nans), data_scanning[~nans])
                
                x = np.arange(len(data_scanning)) 

                # setup fit parameters
                # setup fit parameters
                params = Parameters()
                params.add('timescale', value=0.05,min=0,max=1)
                if 'downup' in alias:
                    params.add('gain', value=0.75)
                elif 'updown' in alias:
                    params.add('gain', value=1.1)

                # these are just for prescription
                params.add('startpoint', value=0)
                params.add('endpoint', value=0)

                # perform fit
                res_react=minimize(model, params, kws={'data':data_react,'x':x})
                res_scanning=minimize(model, params, kws={'data':data_scanning,'x':x})

                # and get differences
                timescale_diff[direction].append(res_react.params['timescale'].value-res_scanning.params['timescale'].value)
                startpoint_diff[direction].append(res_react.params['startpoint'].value-res_scanning.params['startpoint'].value)
                endpoint_diff[direction].append(res_react.params['endpoint'].value-res_scanning.params['endpoint'].value)
        

        f = pl.figure(figsize=(1.2,2.75))
        for di,d in enumerate(['down','up']):
            s = f.add_subplot(2,1,di+1)

            data = np.array(timescale_diff[d])

            p = self.CU.p_val_from_bootstrap_dist(data)
            sns.kdeplot(data,color='k',alpha=0.25,shade=True,lw=1.5)

            pl.axvline(0,color='k',linewidth=0.25)

            if di == 1:
                pl.xlabel('timescale')
                pl.xticks([0,0.5])
            else:
                pl.xticks([])         

            pl.xlim([0,0.5])
            pl.text(0.2,s.get_ylim()[1]*0.75,'p:'+('%.3f'%p)[1:])

            pl.ylabel('density (a.u.)')

            sns.despine(offset=2)
            pl.yticks([0,s.get_ylim()[1]])

        pl.tight_layout()
        pl.savefig(os.path.join(figdir,'timescale_diffs.pdf'%()))
        pl.close()    

    def plot_block_jumps(self):

        if not hasattr(self, 'all_data'):
            self.get_data()

        figdir = os.path.join(self.plot_dir,'Fig4')
        if not os.path.isdir(figdir): os.mkdir(figdir)

        f = pl.figure(figsize=(2.4,1.5))
        for aii, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
            s = f.add_subplot(1,2,aii+1)
            pl.title(['down-to-up' if 'downup' in alias_combo[0] else 'up-to-down'][0])

            for ai, alias in enumerate(alias_combo):
                color = ['orange','c'][np.where(alias.split('_')[0]==np.array(['react','scanning']))[0][0]]

                pl.axhline(0,color='k',ls='--',lw=0.5)

                jumps = []
                for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]):
                    
                    # get block 1 data
                    ma_y1 = self.all_data[alias][subject][1]['ma_y']
                    # get block 2 data
                    ma_y2 = self.all_data[alias][subject][2]['ma_y']

                    # compute jump
                    end_block_1 = np.nanmedian(ma_y1[-1])
                    start_block2 = np.nanmedian(ma_y2[0])
                    jump = start_block2-end_block_1

                    jumps.append(jump)
               
                jumps = np.array(jumps)
                jump_mean = np.nanmean(jumps)
                ci = sms.DescrStatsW(jumps[~np.isnan(jumps)]).tconfint_mean()

                t,p = sp.stats.ttest_1samp(jumps[~np.isnan(jumps)],0)
                if p >.05:
                    fill_color = 'w'
                    edge_color = color
                else:
                    fill_color = color
                    edge_color = 'w'

                pl.plot([ai,ai],ci,color=color,lw=2)
                pl.plot(ai,jump_mean,'o',color=fill_color,mec=edge_color)
            
            # pl.yticks([])
            pl.xlim(-1,2)
            pl.xticks([])
            if aii == 0:
                pl.yticks([-.16,-.08,0,.08,.16])
                pl.ylabel('block 2-3 gain jump')
            else:
                pl.yticks([])
            pl.ylim(-0.16,.16)

            # from matplotlib.ticker import MaxNLocator
            # s.yaxis.set_major_locator(MaxNLocator(5))
            # pl.locator_params(axis='y', nticks=6)
            sns.despine(offset=2)

        pl.tight_layout()
        pl.savefig(os.path.join(figdir,'jumps.pdf'))
        pl.close()


        f = pl.figure(figsize=(1,1.5))
        s = f.add_subplot(1,1,1)
        for aii, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
            # now for the diffs in jumps
            pl.title(['down-to-up' if 'downup' in alias_combo[0] else 'up-to-down'][0])
            pl.axhline(0,color='k',ls='--',lw=0.5)

            jumps = []
            for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]):
                sub_jumps = []
                for ai, alias in enumerate(alias_combo):

                    # get block 1 data
                    ma_y1 =  self.all_data[alias][subject][1]['ma_y']

                    # get block 2 data
                    ma_y2 =  self.all_data[alias][subject][2]['ma_y']

                    # now get jump between blocks
                    end_block_1 = np.nanmean(ma_y1[-1])
                    start_block2 = np.nanmean(ma_y2[0])
                    sub_jumps.append(start_block2-end_block_1)

                jumps.append(sub_jumps[0]-sub_jumps[1]) 

            jumps = np.array(jumps)#.T[0]
            jump_mean = np.nanmean(jumps)
            ci = sms.DescrStatsW(jumps[~np.isnan(jumps)]).tconfint_mean()

            t,p = sp.stats.ttest_1samp(jumps[~np.isnan(jumps)],0)
            d = np.nanmean(jumps)/np.nanstd(jumps)

            if p >.05:
                fill_color = 'w'
                edge_color = 'k'
            else:
                fill_color = 'k'
                edge_color = 'w'
            pl.ylim(-0.16,.16)

            if aii == 0:
                print 'down-up t(%d) = %.3f, p = %.3f, cohen_d = %.3f'%(np.sum(~np.isnan(jumps))-1,t,p,d)
                pl.text(aii,.12,'p:%s'%(('%.3f'%p)[1:]),horizontalalignment='center')
            else:
                print 'up-down t(%d) = %.3f, p = %.3f, cohen_d = %.3f'%(np.sum(~np.isnan(jumps))-1,t,p,d)
                pl.text(aii,-.14,'p:%s'%(('%.3f'%p)[1:]),horizontalalignment='center')

            pl.plot([aii,aii],ci,color='k',lw=2)
            pl.plot(aii,jump_mean,'o',color=fill_color,mec=edge_color)

            pl.xlim(-1,2)
            pl.xticks([])

            pl.yticks([])

            sns.despine(offset=2)

        pl.tight_layout()#w_pad = 0)
        pl.savefig(os.path.join(figdir,'jump_diffs.pdf'))
        pl.close()

    def fit_smith_model(self,block_idx = 1,
        fit_learning_parameters = False
        ):

        if not hasattr(self,'all_data'):
            self.get_data()

        self.all_fit_results = {}
        for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]):
            self.all_fit_results[subject] = {}
            for alias in self.aliases:
                self.all_fit_results[subject][alias] = {}

                # get the relevant data
                data = self.all_data[alias][subject][block_idx]['data']-1

                # determine adaptataion goal 
                if 'updown' in alias:
                    adaptation_goal = self.up_gain-1
                elif 'downup' in alias:
                    adaptation_goal = self.down_gain-1

                print 'now scaling smith model for %s %s, block %s'%(subject,alias,block_idx)

                # slow_learn = .02
                # slow_forget = .992
                # fast_learn = .21
                # fast_forget = .59

                # setup fit parameters
                params = Parameters()
                params.add('slow_learn',value=.02,min=0,max=1,vary=fit_learning_parameters)
                params.add('delta_learn',value=.19,min=0,max=1,vary=fit_learning_parameters)
                params.add('fast_forget',value=.59,min=0,max=1,vary=fit_learning_parameters)
                params.add('delta_forget',value=.402,min=0,max=1,vary=fit_learning_parameters)

                # gain parameters
                params.add('fast_gain', value=1,min=0)
                params.add('slow_gain', value=1,min=0)

                # perform fit
                res=minimize(smith_model, params, kws=
                    {'data':data,
                    'adaptation_goal':adaptation_goal,
                    'fit':True})

                # recreat results
                adaptation = smith_model(res.params,data,adaptation_goal,fit=False)

                # save results
                self.all_fit_results[subject][alias]['adapt'] = adaptation
                self.all_fit_results[subject][alias]['fit_res'] = res
                    
    def plot_fit_results(self,fit_learning_parameters=False,block_idx = 1):

        self.fit_smith_model(fit_learning_parameters=fit_learning_parameters)

        if fit_learning_parameters:
            postFix = '_free_params'
        else:
            postFix = ''

        figdir = os.path.join(self.plot_dir,'Fig5')
        if not os.path.isdir(figdir): os.mkdir(figdir)  
        
        # now start plotting fit results
        fast_color = 'b'
        slow_color = 'k'

        all_subs_data = {}
        for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]+['avg']):
        # for subi,subject in enumerate(['avg']):
            for ac, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
                f = pl.figure(figsize=(2.25,3))
                k=0
                all_subs_data[subject]={}
                for ai, alias in enumerate(alias_combo):
                    all_subs_data[subject][alias] = {}
                    if 'downup' in alias:
                        color = 'g'
                    else:
                        color = 'r'

                    k+=1
                    s = f.add_subplot(2,1,k)
                    pl.title(alias)
                    pl.axhline(1,color='k',ls='-',lw=0.5)
                    # pl.title(alias.split('_')[0])
                    if subject != 'avg':
                        
                        this_block_fit_both = (self.all_fit_results[subject][alias]['adapt']['both']+1)
                        this_block_fit_fast = (self.all_fit_results[subject][alias]['adapt']['fast']+1)
                        this_block_fit_slow = (self.all_fit_results[subject][alias]['adapt']['slow']+1)
                        
                        x_data = np.array(self.all_data[alias][subject][block_idx]['ma_x'])
                        y_data = np.array(self.all_data[alias][subject][block_idx]['ma_y'])

                        pl.plot(x_data,y_data,lw=1.5,c=color)
                        pl.plot(this_block_fit_both,c=color,lw=1.5)
                        pl.plot(this_block_fit_fast,c=fast_color,lw=1.5)
                        pl.plot(this_block_fit_slow,c=slow_color,lw=1.5)

                    else:  

                        x_data = np.array([self.all_data[alias][sub.initials][block_idx]['ma_x'] for sub in self.subjects])
                        y_data = np.array([self.all_data[alias][sub.initials][block_idx]['ma_y'] for sub in self.subjects])
                        fit_both = np.array([(self.all_fit_results[sub.initials][alias]['adapt']['both']+1) for sub in self.subjects])
                        fit_slow = np.array([(self.all_fit_results[sub.initials][alias]['adapt']['slow']+1) for sub in self.subjects])
                        fit_fast = np.array([(self.all_fit_results[sub.initials][alias]['adapt']['fast']+1) for sub in self.subjects])
                        this_block_x = np.nanmean(x_data,axis=0)
                        this_block_y = np.nanmean(y_data,axis=0)
                        this_block_se = np.nanstd(y_data,axis=0)/np.sqrt(len(self.subjects)-1)*stats.t.ppf(1-0.025, len(self.subjects)-1)
                        this_block_fit_both = np.nanmean(fit_both,axis=0)
                        this_block_fit_se_both = np.nanstd(fit_both,axis=0)/np.sqrt(len(self.subjects)-1)*stats.t.ppf(1-0.025, len(self.subjects)-1)
                        this_block_fit_slow = np.nanmean(fit_slow,axis=0)
                        this_block_fit_se_slow = np.nanstd(fit_slow,axis=0)/np.sqrt(len(self.subjects)-1)*stats.t.ppf(1-0.025, len(self.subjects)-1)
                        this_block_fit_fast = np.nanmean(fit_fast,axis=0)
                        this_block_fit_se_fast = np.nanstd(fit_fast,axis=0)/np.sqrt(len(self.subjects)-1)*stats.t.ppf(1-0.025, len(self.subjects)-1)                        

                        # and plot these results
                        pl.fill_between(this_block_x,np.ravel([this_block_y+this_block_se]),np.ravel([this_block_y-this_block_se]),color=color,alpha=.1)
                        pl.fill_between(np.arange(len(this_block_fit_both)),this_block_fit_both+this_block_fit_se_both,this_block_fit_both-this_block_fit_se_both,color=color,alpha=.3)
                        pl.fill_between(np.arange(len(this_block_fit_slow)),this_block_fit_slow+this_block_fit_se_slow,this_block_fit_slow-this_block_fit_se_slow,color=slow_color,alpha=.3)
                        pl.fill_between(np.arange(len(this_block_fit_fast)),this_block_fit_fast+this_block_fit_se_fast,this_block_fit_fast-this_block_fit_se_fast,color=fast_color,alpha=.3)

                        pl.plot(this_block_fit_both,c=color,lw=1.5,ls='-')
                        pl.plot(this_block_fit_slow,c=slow_color,lw=1.5,ls='--')
                        pl.plot(this_block_fit_fast,c=fast_color,lw=1.5,ls='--')
                        
                    if subject == 'avg':
                        if 'updown' in alias:
                            ylims = [1,1.12]
                        else:
                            ylims = [.8,1]
                    else:
                        ylims = [.5,1.5]
                    pl.ylim(ylims)
                    pl.yticks([ylims[0],1,ylims[1]])
                    sns.despine(offset=10)

                    pl.xlim(0,self.n_trials_per_block[block_idx])

                    pl.ylim(ylims)
                    pl.yticks([ylims[0],1,ylims[1]])
                    sns.despine(offset=2)
                    pl.xticks([0,self.n_trials_per_block[block_idx]])
                    pl.ylabel('gain')
                    if ai == 1:
                        pl.xlabel('trial # within block 1')
                
                f.tight_layout()
                if sub == 'avg':
                    f.savefig(os.path.join(figdir,'%s_adaptation_sub_%s%s.pdf'%(alias_combo[0].split('_')[-1],subject,postFix)))
                pl.close()

    def compare_fit_gains(self,fit_learning_parameters=False):

        self.fit_smith_model(fit_learning_parameters=fit_learning_parameters)

        figdir = os.path.join(self.plot_dir,'Fig6')
        if not os.path.isdir(figdir): os.mkdir(figdir)  

        if fit_learning_parameters:
            postFix = '_free_params'
        else:
            postFix = ''

        ##############################################################################
        # get the gains together in one dict (and df for aov)
        ##############################################################################
        block_idx = 1
        # first get gains
        gains_df = {key: [] for key in ['subject','condition','direction','data','process']}
        gains = {}
        for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]):
            gains[subject]={}
            for alias in self.aliases:
                gains[subject][alias] = {}
                for process in ['fast','slow']:
                    this_gain = self.all_fit_results[subject][alias]['fit_res'].params['%s_gain'%(process)].value
                    gains[subject][alias][process] = this_gain
                    gains_df['subject'].append(subi)
                    gains_df['condition'].append(alias.split('_')[0])
                    if 'updown' in alias:
                        gains_df['direction'].append('up')
                    elif 'downup' in alias:
                        gains_df['direction'].append('down')
                    gains_df['data'].append(this_gain)
                    gains_df['process'].append(process)

        ##############################################################################
        # perform the anova
        ##############################################################################

        # let's put it in a df

        # exclude participant if gain parameter > 10
        gains_df_pd = pd.DataFrame(gains_df)
        invalid_subjects = np.where(np.invert(np.array([np.max( np.array(gains_df['data'])[np.array(gains_df['subject'])==i]) for i in range(len(self.subjects))])<10))[0]#==False)
        
        # remove from gains_df
        for i in invalid_subjects:
            gains_df_pd = gains_df_pd[gains_df_pd.subject != i] 

        # and from dict
        invalid_subject_str = np.array([s.initials for s in self.subjects])[invalid_subjects]
        for s in invalid_subject_str:
            del gains[s]
        
        import pyvttbl as pt
        df = pt.DataFrame(gains_df)
        aov = df.anova('data', sub='subject', wfactors=['condition','direction','process'])
        print(aov)

        ##############################################################################
        # plot an overview of different gains in all conditions (plot A)
        ##############################################################################

        x_offset = .15
        # now plot 
        f = pl.figure(figsize=(4,2))
        for ai, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
            s = f.add_subplot(1,2,ai+1)
            if 'downup' in alias_combo[0]:
                pl.title('gain down')
            elif 'updown' in alias_combo[0]:
                pl.title('gain up')

            for aii,alias in enumerate(alias_combo):

                # get slow gains
                slow_gains = [gains[sub][alias]['slow'] for sub in gains.keys()]
                # detect_outliers:


                slow_mean = np.mean(slow_gains)
                slow_ci=sms.DescrStatsW(slow_gains).tconfint_mean()

                # get fast gains
                fast_gains = [gains[sub][alias]['fast'] for sub in gains.keys()]
                fast_mean = np.mean(fast_gains)
                fast_ci=sms.DescrStatsW(fast_gains).tconfint_mean()

                j = aii*x_offset
                slow_offsets = np.linspace(j-x_offset/4,j+x_offset/4,len(slow_gains))
                fast_offsets = np.linspace(1+j-x_offset/4,1+j+x_offset/4,len(fast_gains))
                for si in range(len(gains.keys())):
                    pl.plot([slow_offsets[si],fast_offsets[si]],[slow_gains[si],fast_gains[si]],ms=0.2,marker=['o','s'][aii],color=['orange','c'][aii],lw=0.5,alpha=0.2,mec='w')
                pl.plot(slow_offsets,slow_gains,'o',ms=2,color=['orange','c'][aii],alpha=0.5,mec='w')
                pl.plot(fast_offsets,fast_gains,'o',ms=2,color=['orange','c'][aii],alpha=0.5,mec='w')

                # plot
                pl.plot([0+j,0+j],slow_ci,color='w',lw=2)
                pl.plot([0+j,0+j],slow_ci,color=['orange','c'][aii],lw=1.5)
                pl.plot([1+j,1+j],fast_ci,color='w',lw=2)
                pl.plot([1+j,1+j],fast_ci,color=['orange','c'][aii],lw=1.5)
                pl.plot([0+j,1+j],[slow_mean,fast_mean],marker=['o','s'][aii],color='w',lw=2,mec='w')
                pl.plot([0+j,1+j],[slow_mean,fast_mean],marker=['o','s'][aii],color=['orange','c'][aii],lw=1.5,label=alias.split('_')[0],mec='w')

                if ai == 1:
                    pl.legend()

            sns.despine(offset=10)
            pl.ylabel('gain')
            pl.xticks([0+x_offset/2,1+x_offset/2],['slow','fast'])
            pl.xlim(-0.25,1.3)
            if fit_learning_parameters:
                pl.ylim([-0.02,4.8])
                pl.yticks([0,1.2,2.4,3.6,4.8],['0','1.2','3.6','2.4','4.8'])                
            else:
                pl.ylim([-0.02,1.6])
                pl.yticks([0,0.4,0.8,1.2,1.6],['0','0.4','0.8','1.2','1.6'])

        f.tight_layout()
        f.savefig(os.path.join(figdir,'gains%s.pdf'%postFix))
        pl.close()

        ##############################################################################
        # differential process contribution to both conditions (fig B)
        ##############################################################################

        processes = ['fast','slow']
        for ratio in [True]:
            
            x_offset = .2
            f = pl.figure(figsize=(2,2))
            s = f.add_subplot(1,1,1)
            conditions = ['react','scanning']

            for pi, condition in enumerate(conditions):            
               
                up_total = np.sum([np.array([gains[sub]['%s_updown'%condition][p] for sub in gains.keys()]) for p in processes],axis=0)
                down_total = np.sum([np.array([gains[sub]['%s_downup'%condition][p] for sub in gains.keys()]) for p in processes],axis=0)

                # if ratio:

                # get slow gains
                up_slow = np.array([gains[sub]['%s_updown'%condition]['slow'] for sub in gains.keys()])
                down_slow = np.array([gains[sub]['%s_downup'%condition]['slow'] for sub in gains.keys()])
                if ratio:
                    up_slow /= up_total
                    down_slow /= down_total
                slow_gains = np.mean([up_slow,down_slow],axis=0)

                # get fast gains
                up_fast = np.array([gains[sub]['%s_updown'%condition]['fast'] for sub in gains.keys()])
                down_fast = np.array([gains[sub]['%s_downup'%condition]['fast'] for sub in gains.keys()])
                if ratio:
                    up_fast /= up_total
                    down_fast /= down_total
                fast_gains = np.mean([up_fast,down_fast],axis=0)

                # then get difference
                if ratio:
                    gain_diffs = slow_gains
                    test_value = 0.5
                else:
                    gain_diffs = slow_gains-fast_gains
                    test_value = 0
                
                pl.axhline(test_value,lw=0.5,color='k')

                gain_diff_mean = np.mean(gain_diffs)
                gain_diff_ci=sms.DescrStatsW(gain_diffs).tconfint_mean()

                t,p = sp.stats.ttest_1samp(gain_diffs,test_value)
                if ratio:
                    cohen_d = np.mean(gain_diffs-0.5)/np.std(gain_diffs)       
                else:
                    cohen_d = np.mean(gain_diffs)/np.std(gain_diffs)       

                pl.text(pi,.12,'p:%s'%(('%.4f'%p)[1:]),horizontalalignment='center')

                if p < .05:
                    fill_color = ['orange','c'][pi]
                    edge_color= 'w'
                else:
                    fill_color = 'w'
                    edge_color= ['orange','c'][pi]

                if ratio:
                    print 'slow gain ratio %s = %.3f, t(%d)= %.3f, p = %.3E, d = %.3f'%(condition,np.mean(gain_diffs),len(gain_diffs)-1,t,Decimal(p),cohen_d)
                else:
                   print 'slow-fast gain %s = %.3f, t(%d)= %.3f, p = %.3E, d = %.3f'%(condition,np.mean(gain_diffs),len(gain_diffs)-1,t,Decimal(p),cohen_d)

                pl.plot(np.linspace(pi-x_offset,pi+x_offset,len(gain_diffs)),gain_diffs,'o',ms=3,color=['orange','c'][pi],mec='w',alpha=0.5)
                # plot
                pl.plot([pi,pi],gain_diff_ci,color='w',lw=3)
                pl.plot([pi,pi],gain_diff_ci,color=['orange','c'][pi],lw=2)
                pl.plot([pi],[gain_diff_mean],marker=['o','s'][pi],ms=7.5,color=fill_color,label=process,mec=edge_color)

                sns.despine(offset=10)
                pl.xticks([0,1],conditions)
                pl.xlim(-x_offset*2,1+x_offset*2)


                if ratio:
                    pl.ylabel('slow gain ratio')
                    pl.yticks([0,1])
                    pl.ylim([0,1])
                else:
                    pl.ylabel(r'slow-fast gain')
                    pl.ylim([-1,1])
                    pl.yticks([-1,1])

            pl.tight_layout()
            f.savefig(os.path.join(figdir,'slow_fast_gain_comparison_collapsed_updown_ratio_%s%s.pdf'%(ratio,postFix)))
            pl.close()            

        ##############################################################################
        # difference in react-scanning gain for both slow and fast process (fig C)
        ##############################################################################

        for ratio in [False]:

            x_offset = .2
            f = pl.figure(figsize=(2,2))
            s = f.add_subplot(1,1,1)
            pl.axhline(0,lw=0.5,color='k')
            processes = ['slow','fast']

            react_up_total = np.sum([np.array([gains[sub]['react_updown'][p] for sub in gains.keys()]) for p in processes],axis=0)
            react_down_total = np.sum([np.array([gains[sub]['react_downup'][p] for sub in gains.keys()]) for p in processes],axis=0)
            scanning_up_total = np.sum([np.array([gains[sub]['scanning_updown'][p] for sub in gains.keys()]) for p in processes],axis=0)
            scanning_down_total = np.sum([np.array([gains[sub]['scanning_downup'][p] for sub in gains.keys()]) for p in processes],axis=0)

            for pi,process in enumerate(processes):

                # get react gains
                react_up = np.array([gains[sub]['react_updown'][process] for sub in gains.keys()])
                react_down = np.array([gains[sub]['react_downup'][process] for sub in gains.keys()])
                if ratio:
                    react_up/=react_up_total
                    react_down/=react_down_total
                react_gains = np.mean([react_up,react_down],axis=0)

                # get scanning gains
                scanning_up = np.array([gains[sub]['scanning_updown'][process] for sub in gains.keys()])
                scanning_down = np.array([gains[sub]['scanning_downup'][process] for sub in gains.keys()])
                if ratio:
                    scanning_up/=scanning_up_total
                    scanning_down/=scanning_down_total
                scanning_gains = np.mean([scanning_up,scanning_down],axis=0)

                # then get difference
                gain_diffs = react_gains-scanning_gains
                gain_diff_mean = np.mean(gain_diffs)
                gain_diff_ci=sms.DescrStatsW(gain_diffs).tconfint_mean()

                t,p = sp.stats.ttest_1samp(gain_diffs,0)
                cohen_d = np.mean(gain_diffs)/np.std(gain_diffs)
                print 'react - scanning %s gain ratio %s = %.3f, t(%d)= %.3f, p = %.3E,d = %.3f'%(process,ratio,np.mean(gain_diffs),len(gain_diffs)-1,t,Decimal(p),cohen_d)
                if p < .05:
                    fill_color = ['b','k'][pi]
                    edge_color= 'w'
                else:
                    fill_color = 'w'
                    edge_color= ['b','k'][pi]

                # if pi == 0:
                pl.text(pi,-0.76,'p:%s'%(('%.3f'%p)[1:]),horizontalalignment='center')
                # else:
                    # pl.text(pi,-0.5,'p:%s'%(('%.4f'%p)[1:]),horizontalalignment='center')

                pl.plot(np.linspace(pi-x_offset,pi+x_offset,len(gain_diffs)),gain_diffs,'o',ms=3,color=['b','k'][pi],mec='w',alpha=0.5)
                # plot
                pl.plot([pi,pi],gain_diff_ci,color='w',lw=3)
                pl.plot([pi,pi],gain_diff_ci,color=['b','k'][pi],lw=2)
                pl.plot([pi],[gain_diff_mean],marker=['o','s'][pi],ms=7.5,color=fill_color,label=process,mec=edge_color)
                # x = np.argmax(np.max())

                sns.despine(offset=10)
                pl.xticks([0,1],processes)
                pl.xlim(-x_offset*2,1+x_offset*2)
                if fit_learning_parameters:
                    pass
                else:
                    pl.ylim([-1,1])
                    pl.yticks([-1,1])

                pl.ylabel(r'react - scanning gain')

            pl.tight_layout()
            f.savefig(os.path.join(figdir,'react_scanning_gain_comparison_collapsed_updown_ratio_%s%s.pdf'%(ratio,postFix)))
            pl.close()

        ##############################################################################
        # slow ratio difference between up and down adaptation (not in paper, but for stats)
        ##############################################################################

        f = pl.figure(figsize=(4,2))
        x_offset = .2
        cl='gray'
        for ri,ratio in enumerate([False,True]):
            
            s = f.add_subplot(1,2,1+ri)   
            pl.axhline(0,lw=0.5,color='k')

            # get react up fast ratio
            react_up_fast_gains = np.array([gains[sub]['react_updown']['fast'] for sub in gains.keys()])
            react_up_slow_gains = np.array([gains[sub]['react_updown']['slow'] for sub in gains.keys()])
            if ratio:
                react_up_fast_ratio = react_up_fast_gains/(react_up_slow_gains+react_up_fast_gains)
            else:
                react_up_fast_ratio = react_up_fast_gains - react_up_slow_gains

            # get react up fast ratio
            scanning_up_fast_gains = np.array([gains[sub]['scanning_updown']['fast'] for sub in gains.keys()])
            scanning_up_slow_gains = np.array([gains[sub]['scanning_updown']['slow'] for sub in gains.keys()])
            scanning_up_fast_ratio = scanning_up_fast_gains/(scanning_up_slow_gains+scanning_up_fast_gains)
            
            if ratio:
                scanning_up_fast_ratio = scanning_up_fast_gains/(scanning_up_slow_gains+scanning_up_fast_gains)
            else:
                scanning_up_fast_ratio = scanning_up_fast_gains - scanning_up_slow_gains

            # combine over conditions
            up_fast_ratio = np.mean([react_up_fast_ratio,scanning_up_fast_ratio],axis=0)

            # get react down fast ratio
            react_down_fast_gains = np.array([gains[sub]['react_downup']['fast'] for sub in gains.keys()])
            react_down_slow_gains = np.array([gains[sub]['react_downup']['slow'] for sub in gains.keys()])
            react_down_fast_ratio = react_down_fast_gains/(react_down_slow_gains+react_down_fast_gains)
            if ratio:
                react_down_fast_ratio = react_down_fast_gains/(react_down_slow_gains+react_down_fast_gains)
            else:
                react_down_fast_ratio = react_down_fast_gains - react_down_slow_gains

            # get scanning down fast ratio
            scanning_down_fast_gains = np.array([gains[sub]['scanning_downup']['fast'] for sub in gains.keys()])
            scanning_down_slow_gains = np.array([gains[sub]['scanning_downup']['slow'] for sub in gains.keys()])
            scanning_down_fast_ratio = scanning_down_fast_gains/(scanning_down_slow_gains+scanning_down_fast_gains)
           
            if ratio:
                scanning_down_fast_ratio = scanning_down_fast_gains/(scanning_down_slow_gains+scanning_down_fast_gains)
            else:
                scanning_down_fast_ratio = scanning_down_fast_gains - scanning_down_slow_gains
            

            # now mean over scanning and react
            down_fast_ratio = np.mean([react_down_fast_ratio,scanning_down_fast_ratio],axis=0)

            # up-down fast ratios
            up_down_fast_ratios = up_fast_ratio-down_fast_ratio
            if ratio:
                cohen_d = np.mean(up_down_fast_ratios-0.5)/np.std(up_down_fast_ratios)
            else:
                cohen_d = np.mean(up_down_fast_ratios)/np.std(up_down_fast_ratios)

            gain_diff_mean = np.mean(up_down_fast_ratios)
            gain_diff_ci=sms.DescrStatsW(up_down_fast_ratios).tconfint_mean()
            t,p = sp.stats.ttest_1samp(up_down_fast_ratios,0)
            
            if ratio:
                print 'fast/(fast+slow) for up-down gain  = %.3f, t(%d)= %.3f, p = %.3E,d = %.3f'%(np.mean(up_down_fast_ratios),len(up_down_fast_ratios),t,Decimal(p),cohen_d)
            else:
                print 'fast-slow for up-down gain = %.3f, t(%d)= %.3f, p = %.3E,d = %.3f'%(np.mean(up_down_fast_ratios),len(up_down_fast_ratios),t,Decimal(p),cohen_d)

            if p < .05:
                fill_color = cl
                edge_color = 'w'
            else:
                fill_color = 'w'
                edge_color = cl

            ci = 0
            pl.plot(np.linspace(ci-x_offset,ci+x_offset,len(up_down_fast_ratios)),up_down_fast_ratios,'o',ms=3,color=cl,mec='w',alpha=0.5)
            # plot
            pl.plot([ci,ci],gain_diff_ci,color='w',lw=3)
            pl.plot([ci,ci],gain_diff_ci,color=cl,lw=2)
            pl.plot([ci],[gain_diff_mean],marker='o',ms=7.5,color=fill_color,mec=edge_color)

            sns.despine(offset=10)
            pl.xticks([])
            pl.xlim(-x_offset*2,x_offset*2)
            pl.ylim([-1,1])
            pl.yticks([-1,1])

            if ratio:
                pl.ylabel(r'up-down fast ratio')
            else:
                pl.ylabel(r'up-down fast-slow')

        pl.tight_layout()
        f.savefig(os.path.join(figdir,'fast_ratio_updowndiff%s.pdf'%postFix))
        pl.close()


    def plot_seen_reports(self):

        if not hasattr(self,'all_data'):
            self.get_data()

        figdir = os.path.join(self.plot_dir,'Fig7')
        if not os.path.isdir(figdir): os.mkdir(figdir)  

        ##############################################################################
        # first plot all blocks
        ##############################################################################
        all_subs_data = {}

        ylims = [0,1]
        for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]+['avg']):
        # for subi,subject in enumerate(['avg']):
            all_subs_data[subject] = {}

            for aii, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):
                f = pl.figure(figsize=(3,2))
                for ai, alias in enumerate(alias_combo):
                    all_subs_data[subject][alias] = {}
                    s = f.add_subplot(2,1,ai+1)
                    # pl.axhline(1,color='k',ls='--',lw=0.5)

                    if 'downup' in alias:
                        colors = ['k','g','r','k']
                    else:
                        colors = ['k','r','g','k']

                    for bi in range(self.nr_blocks):
                        all_subs_data[subject][alias][bi] = {}

                        tp = np.hstack([0,np.cumsum(self.n_trials_per_block)])[bi]
                        pl.axvline(tp,color='k',linewidth=0.5)                        

                        if subject == 'avg':
                            x_data = np.array([all_subs_data[sub.initials][alias][bi]['ma_x'] for sub in self.subjects])
                            y_data = np.array([all_subs_data[sub.initials][alias][bi]['ma_y'] for sub in self.subjects])
                            this_block_x = np.nanmean(x_data,axis=0)
                            this_block_y = np.nanmean(y_data,axis=0)
                            this_block_se = np.nanstd(y_data,axis=0)/np.sqrt(len(self.subjects)-1)*stats.t.ppf(1-0.025, len(self.subjects)-1)
                            pl.fill_between(this_block_x,np.ravel([this_block_y+this_block_se]),np.ravel([this_block_y-this_block_se]),color=colors[bi],alpha=.2)
                            pl.plot(this_block_x,this_block_y,color=colors[bi],lw=1)
                        else:
                            this_block_data = self.all_data[alias][subject][bi]['seen']                            
                            # moving average
                            ma_steps = self.n_directions
                            this_block_data_padded = np.hstack([np.ones(ma_steps)*np.nan,this_block_data,np.ones(ma_steps)*np.nan])
                            ma_y_se = np.array([np.nanstd(this_block_data_padded[i:i+ma_steps])/np.sqrt(ma_steps-1) for i in range(len(this_block_data_padded)-ma_steps)])
                            ma_y = np.array([np.nanmean(this_block_data_padded[i:i+ma_steps]) for i in range(len(this_block_data_padded)-ma_steps)])
                            ma_x = np.linspace(self.block_trial_indices[bi][0],self.block_trial_indices[bi][-1],len(ma_y))
                            all_subs_data[subject][alias][bi]['ma_y'] = ma_y
                            all_subs_data[subject][alias][bi]['ma_x'] = ma_x

                            # plot 
                            pl.fill_between(ma_x,ma_y+ma_y_se,ma_y-ma_y_se,color=colors[bi],alpha=.2)
                            pl.plot(ma_x,ma_y,c=colors[bi],lw=1)
                        pl.ylim(ylims)

                        if ai ==1:
                            pl.xticks(np.hstack([0,np.cumsum(self.n_trials_per_block)]))
                        else:
                            pl.xticks([])
                        
                        # if bi == 0:
                        pl.yticks([ylims[0],1,ylims[1]])                            
                        # else:
                            # pl.yticks([])

                        if (ai==1):
                            pl.ylabel('seen')
                            pl.xlabel('trial #')

                        sns.despine(offset=2)

                pl.tight_layout()
                if subject == 'avg':
                    pl.savefig(os.path.join(figdir,'%s_seen_sub_%s.pdf'%(alias_combo[0].split('_')[-1],subject)))
                pl.close()

        ##############################################################################
        # only plot first block
        ##############################################################################

        block_idx = 1
        x_offset = .15
        # now plot
        f = pl.figure(figsize=(2,2))
        s = f.add_subplot(1,1,1)
        for ai, alias_combo in enumerate([['react_downup','scanning_downup'],['react_updown','scanning_updown']]):

            # get up gains
            react_seens = [np.mean(self.all_data[alias_combo[0]][sub.initials][block_idx]['seen']) for sub in self.subjects]
            react_mean = np.mean(react_seens)
            react_ci=sms.DescrStatsW(react_seens).tconfint_mean()

            # get fast seens
            scanning_seens = [np.mean(self.all_data[alias_combo[1]][sub.initials][block_idx]['seen']) for sub in self.subjects]
            scanning_mean = np.mean(scanning_seens)
            scanning_ci=sms.DescrStatsW(scanning_seens).tconfint_mean()

            j = ai*x_offset
            pl.plot(np.linspace(j-x_offset/4,j+x_offset/4,len(react_seens)),react_seens,'o',mec='w',ms=3,color=['g','r'][ai],alpha=0.5)
            pl.plot(np.linspace(1+j-x_offset/4,1+j+x_offset/4,len(scanning_seens)),scanning_seens,'o',mec='w',ms=3,color=['g','r'][ai],alpha=0.5)

            # plot
            pl.plot([0+j,0+j],react_ci,color=['g','r'][ai],lw=1.5)
            pl.plot([1+j,1+j],scanning_ci,color=['g','r'][ai],lw=1.5)
            pl.plot([0+j,1+j],[react_mean,scanning_mean],marker=['o','s'][ai],color=['g','r'][ai],mec='w',label=['down','up'][ai],lw=1.5,ms=7)


            # if ai == 1:
                # pl.legend()

            sns.despine(offset=2)
            pl.xticks([0+x_offset/2,1+x_offset/2],['react','scanning'])
            pl.ylim(0,1)
            pl.yticks([0,1])#['unseen','seen'])
            pl.ylabel('seen')
            pl.xlim(-0.2,1.45)

        f.tight_layout()
        f.savefig(os.path.join(figdir,'seens.pdf'))
        pl.close()


        seen_df = {}
        block_idx = 1
        # first get gains
        seen_df = {key: [] for key in ['subject','condition','direction','data']}
        gains = {}
        for subi,subject in enumerate([self.subjects[si].initials for si in range(len(self.subjects))]):
            for alias in self.aliases:
                this_seen = np.mean(self.all_data[alias][subject][block_idx]['seen'])
                seen_df['subject'].append(subi)
                seen_df['condition'].append(alias.split('_')[0])
                if 'updown' in alias:
                    seen_df['direction'].append('up')
                elif 'downup' in alias:
                    seen_df['direction'].append('down')
                seen_df['data'].append(this_seen)
            
        ##############################################################################
        # perform the anova
        ##############################################################################

        # let's put it in a df
        import pyvttbl as pt
        df = pt.DataFrame(seen_df)
        aov = df.anova('data', sub='subject', wfactors=['condition','direction'])
        print(aov)