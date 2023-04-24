"""Test Study for the exploit model MOL CC SN batch update.

A study to test the runner with the exploit model MOL CC SN.
It includes the module components exploit_social_learning_MOL_CC_SN,
most_simple_vegetation and simple_extraction.
"""

# This file is part of pycopancore.
#
# Copyright (C) 2016-2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license

import numpy as np
from time import time
import datetime as dt
import pandas as pd
from pymofa.experiment_handling import experiment_handling as eh
from numpy import random
import json
import pickle
import itertools as it
import importlib
import os
import pycopancore.models.exploit_MOLCCSN_batchupdate as M
from pycopancore.runners.runner import Runner

experiment_name = "batchupdate2"

SAVE_FOLDER = f"C:/Users/hotz/Documents/Internship_Copan/Results/{experiment_name}" # name of folder or file?
os.mkdir(SAVE_FOLDER) #makes sure that file/folder with same name doesn't exist
print(f"Directory created @ {SAVE_FOLDER}")
SAVE_PATH_RAW = SAVE_FOLDER + "\\" + "raw"
os.mkdir(SAVE_PATH_RAW)
SAVE_PATH_RES = SAVE_FOLDER + "\\" + "res"
os.mkdir(SAVE_PATH_RES)
# create folder with name experiment_name with 2 files in it raw and res? see later in code when used

SAMPLE_SIZE = 2

#####################################################################################################

# MODEL CONFIGURATION

# ---configuration---

###### facts - just for memory

# choice of p_2 etc.

###### actual parameters

timeinterval = 100
timestep = 0.1
ni_sust = 100  # number of agents with sustainable strategy 1
ni_nonsust = 100  # number of agents with unsustainable strategy 0
nindividuals = ni_sust + ni_nonsust
nc = nindividuals # number of cells
p = 0.4  # link density
effort_difference = [0.5] # half of the difference in effort between sus and con farmers 
average_waiting_time = [2]

# cells parameters

stock = 1
capacity = 1
growth_rate = 1

#individuals parameters

past_strategy = 0
past_harvest_rate = 0

# individuals parameters for initially conventional farmer

con_strategy = 0

w_neighbors = [0,1]

con_w_neighbors_sus = w_neighbors
con_w_neighbors_con = w_neighbors

w_ownland = [0,1]

con_w_ownland_sus = w_ownland
con_w_ownland_con = w_ownland

w_norm = [0,1]

con_w_social_norm_sus = w_norm
con_w_social_norm_con = w_norm

con_w_identity_sus = 0
con_w_identity_con = 0

# individuals parameters for initially sustainable farmer

sus_strategy = 1

sus_w_neighbors_sus = w_neighbors
sus_w_neighbors_con = w_neighbors

sus_w_ownland_sus = w_ownland
sus_w_ownland_con = w_ownland

sus_w_social_norm_sus = w_norm
sus_w_social_norm_con = w_norm

sus_w_identity_sus = 0
sus_w_identity_con = 0

# ---write into dic---

configuration = {"SAMPLE_SIZE":SAMPLE_SIZE,"timeinterval": timeinterval,
    "timestep": timestep,
    "ni_sust" : ni_sust,
    "ni_nonsust" : ni_nonsust,
    "p" : p,
    "effort_difference" : effort_difference,
    "average_waiting_time" : average_waiting_time,
    "stock" : stock,
    "capacity" : capacity,
    "growth_rate" : growth_rate,
    "past_strategy" : past_strategy,
    "past_harvest_rate" : past_harvest_rate,
    "con_strategy" : con_strategy,
    "con_w_neighbors_sus" : con_w_neighbors_sus,
    "con_w_neighbors_con" : con_w_neighbors_con,
    "con_w_ownland_sus" : con_w_ownland_sus,
    "con_w_ownland_con" : con_w_ownland_con,
    "con_w_social_norm_sus" : con_w_social_norm_sus,
    "con_w_social_norm_con" : con_w_social_norm_con,
    "con_w_identity_sus" : con_w_identity_sus,
    "con_w_identity_con" : con_w_identity_con,
    "sus_strategy" : sus_strategy,
    "sus_w_neighbors_sus" : sus_w_neighbors_sus,
    "sus_w_neighbors_con" : sus_w_neighbors_con,
    "sus_w_ownland_sus" : sus_w_ownland_sus,
    "sus_w_ownland_con" : sus_w_ownland_con,
    "sus_w_social_norm_sus" : sus_w_social_norm_sus,
    "sus_w_social_norm_con" : sus_w_social_norm_con,
    "sus_w_identity_sus" : sus_w_identity_sus,
    "sus_w_identity_con" : sus_w_identity_con
    }

# saving config
# ---save json file---
print("Saving config.json")
f = open(SAVE_FOLDER + "\\" + "config.json", "w+")
json.dump(configuration, f, indent=4)
print("Done saving config.json.")

# text file
print("Saving readme.txt.")
with open(SAVE_FOLDER + "\\" + 'readme.txt', 'w') as f:
    f.write('my message here')
print("Done saving readme.txt.")
# überlegen ob ich das brauche

#####################################################################################################

# Defining an experiment execution function according pymofa

     
def RUN_FUNC(average_waiting_time, effort_difference, w_neighbors, w_ownland, w_norm, filename):     
    
    # instantiate model
   model = M.Model()
   
   # instantiate process taxa culture:
   culture = M.Culture(average_waiting_time= average_waiting_time, value_neigh_con = 0,
        value_neigh_sus = 0,
        value_own_con = 0,
        value_own_sus = 0,
        value_norm_con = 0,
        value_norm_sus = 0,
        sus_ind = 0,
        con_ind = 0)
   
   # generate entitites:
   world = M.World(culture=culture)
   social_system = M.SocialSystem(world=world)
   cells = [M.Cell(stock=stock, capacity=capacity, growth_rate=growth_rate, social_system=social_system)
            for c in range(nc)]
   individuals = [M.Individual(strategy = con_strategy, effort_difference=effort_difference , past_strategy =past_strategy,
                               past_harvest_rate = past_harvest_rate,
                               w_neighbors_sus = w_neighbors, w_neighbors_con = w_neighbors,
                               w_ownland_sus =  w_ownland, w_ownland_con = w_ownland,
                               w_social_norm_sus = w_norm, 
                               w_social_norm_con = w_norm, w_identity_sus = con_w_identity_sus, 
                               w_identity_con = con_w_identity_con,
                               cell=cells[i]) for i in range(ni_nonsust)]\
                 + [M.Individual(strategy = sus_strategy, effort_difference=effort_difference, past_strategy =past_strategy,
                                 past_harvest_rate = past_harvest_rate,
                                 w_neighbors_sus = w_neighbors, w_neighbors_con = w_neighbors,
                                             w_ownland_sus =  w_ownland, w_ownland_con = w_ownland,
                                             w_social_norm_sus = w_norm, 
                                             w_social_norm_con = w_norm, w_identity_sus = sus_w_identity_sus,
                                             w_identity_con = sus_w_identity_con,  
                                 cell=cells[i+ni_nonsust])
                    for i in range(ni_sust)]
   
   for (i, c) in enumerate(cells):
       c.individual = individuals[i]
       
   def erdosrenyify(graph, p=0.5):
       """Create a ErdosRenzi graph from networkx graph.
   
       Take a a networkx.Graph with nodes and distribute the edges following the
       erdos-renyi graph procedure.
       """
       assert not graph.edges(), "your graph has already edges"
       nodes = list(graph.nodes())
       for i, n1 in enumerate(nodes[:-1]):
           for n2 in nodes[i+1:]:
               if random.random() < p:
                   graph.add_edge(n1, n2)
   # set the initial graph structure to be an erdos-renyi graph
   print("erdosrenyifying the graph ... ", end="", flush=True)
   start = time()
   erdosrenyify(culture.acquaintance_network, p=p)
   print("done ({})".format(dt.timedelta(seconds=(time() - start))))
   
   print('\n runner starting')
   
   # Define termination signals as list [ signal_method, object_method_works_on ]
   signal_consensus = [M.Culture.check_for_consensus,
                       culture]
   # Define termination_callables as list of all signals
   termination_callables = [signal_consensus]
   print('termination_callables: ', termination_callables)
   
   r = Runner(model=model)
   start = time()
   traj = r.run(t_1=timeinterval, dt=timestep)
   runtime = dt.timedelta(seconds=(time() - start))
   print('runtime: {runtime}'.format(**locals()))
   
   t = np.array(traj['t'])
   print("max. time step", (t[1:]-t[:-1]).max())
   # print('keys:', np.array(traj.keys()))
   # print('completeDict: ', traj)
   
   individuals_strategies = np.array([traj[M.Individual.strategy][ind]
                                    for ind in individuals])
   
   nopinion1_list = np.sum(individuals_strategies, axis=0) / nindividuals
   nopinion0_list = 1 - nopinion1_list
   value_own_con_list =np.array(traj[M.Culture.value_own_con][culture])
   value_own_sus_list =np.array(traj[M.Culture.value_own_con][culture])
   value_neigh_con_list =np.array(traj[M.Culture.value_neigh_con][culture])
   value_neigh_sus_list =np.array(traj[M.Culture.value_neigh_sus][culture])
   value_norm_con_list =np.array(traj[M.Culture.value_norm_con][culture])
   value_norm_sus_list =np.array(traj[M.Culture.value_norm_sus][culture])
   
   prep = {'t':t , 'nopinion1_list':nopinion1_list, 'nopinion0_list':nopinion0_list, 'value_own_con_list':value_own_con_list,
           'value_own_sus_list': value_own_sus_list,
          'value_neigh_con_list' :value_neigh_con_list, 'value_neigh_sus_list':value_neigh_sus_list, 
          'value_norm_con_list' : value_norm_con_list, 'value_norm_sus_list' : value_norm_sus_list}

   res = pd.DataFrame(data=prep)
   res.to_pickle(filename)
   
   # delete old taxa to avoid instantiation errors
   #del model
   world.delete()
   culture.delete()
   social_system.delete()
   for c in cells:
        c.delete()
   for i in individuals:
        i.delete()
       
   # to check if everything went well
   exit_status = 1

   return exit_status

# -----PYMOFA-----

# it.product creates a list of all combinations of params

combis = []
for wn in w_neighbors:
    for wm in w_ownland:
        for wsn in w_norm:
            if wn + wm + wsn == 1:
                combis.append((wn, wm, wsn))
                
combis2 = list(it.product(average_waiting_time, effort_difference, combis))
PARAM_COMBS = [(a,b,c,d,e) for a,b,(c,d,e) in combis2]

#PARAM_COMBS = list(it.product(average_waiting_time, effort_difference, w_neighbors, w_ownland, w_norm))
parameter_name_list = ["average_waiting_time", "effort_difference", "w_neighbors", "w_ownland", "w_norm"]
INDEX = INDEX = {i: parameter_name_list[i] for i in range(len(parameter_name_list))}
# create experiment handle
handle = eh(sample_size=SAMPLE_SIZE, parameter_combinations=PARAM_COMBS, index = INDEX,
            path_raw=SAVE_PATH_RAW, path_res=SAVE_PATH_RES)
# actually run the whole thing
handle.compute(RUN_FUNC)

##### POSTPROCESSING #####

# how to call these results
filename = "stateval_results.pkl"

def sem(fnames):
    """calculate the standard error of the mean for the data in the files
    that are in the list of fnames
    Parameter:
    ----------
    fnames: string
        list of strings of filenames containing simulation results
    Returns:
    sem: float
        Standard error of the mean of the data in the files specified
        by the list of fnames
    """
   # import scipy.stats as st

    return pd.concat([np.load(f, allow_pickle=True) for f in fnames if "traj" not in f]).groupby(level=0).mean()

EVA = {
    "mean": lambda fnames: pd.concat([np.load(f, allow_pickle=True)
                                      for f in fnames if "traj" not in f]).groupby(level=0).mean(),
    "std": lambda fnames: pd.concat([np.load(f, allow_pickle=True)
                                     for f in fnames if "traj" not in f]).groupby(level=0).std()
}

handle.resave(EVA, filename)

#runtime = dt.timedelta(seconds=(time() - start))
#print('runtime: {runtime}'.format(**locals()))

    