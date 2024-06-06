#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:27:29 2021

@author: danielmckenzie

Simple example of using ZORO (without a regularizer).

"""
from optimizers import *
from benchmarkfunctions import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
#mpl.style.use('seaborn')

# problem set up
n = 2000
s = int(0.1*n)
noiseamp = 0.001 # noise amplitude
obj_func = MaxK(n, s, noiseamp)

# Choose initialization                                                         
x0    = np.random.randn(n)
x0    = 100*x0/np.linalg.norm(x0)
xx0   = copy.deepcopy(x0)

sparsity = s
#sparsity = int(0.1*len(x0)) # This is a decent default, if no better estimate is known. 

# Parameters for ZORO. Defaults are fine in most cases
params = {"step_size":1.0, "delta": 0.0001, "max_cosamp_iter": 10, 
          "cosamp_tol": 0.5,"sparsity": sparsity,
          "num_samples": int(np.ceil(np.log(len(x0))*sparsity)), "sampling": "rademacher"}

params_sjlt = {"step_size":1.0, "delta": 0.0001, "max_cosamp_iter": 10, 
          "cosamp_tol": 0.5,"sparsity": sparsity,
          "num_samples": int(np.ceil(np.log(len(x0))*sparsity)), "sampling": "sjlt"}

performance_log_ZORO = [[0, obj_func(x0)]]
performance_log_ZORO_sjlt = [[0, obj_func(x0)]]


# initialize optimizer object
opt  = ZORO(x0, obj_func, params, function_budget= int(1e5))
opt_sjlt  = ZORO(x0, obj_func, params_sjlt, function_budget= int(1e5))


# the actual optimization routine
termination = False
while termination is False:
    # optimization step
    # solution_ZORO = False until a termination criterion is met, in which 
    # case solution_ZORO = the solution found.
    # termination = False until a termination criterion is met.
    # If ZORO terminates because function evaluation budget is met, 
    # termination = B
    # If ZORO terminated because the target accuracy is met,
    # termination= T.
    
    evals_ZORO, solution_ZORO, termination = opt.step()

    # save some useful values
    performance_log_ZORO.append( [evals_ZORO,np.mean(opt.fd)] )
    # print some useful values
    opt.report( 'Estimated f(x_k) using ZORO: %f  function evals: %d\n' %
        (np.mean(opt.fd), evals_ZORO) )
   

termination_sjlt = False
while termination_sjlt is False:
    # optimization step
    # solution_ZORO = False until a termination criterion is met, in which 
    # case solution_ZORO = the solution found.
    # termination = False until a termination criterion is met.
    # If ZORO terminates because function evaluation budget is met, 
    # termination = B
    # If ZORO terminated because the target accuracy is met,
    # termination= T.
    
    evals_ZORO_sjlt, solution_ZORO_sjlt, termination_sjlt = opt_sjlt.step()

    # save some useful values
    performance_log_ZORO_sjlt.append( [evals_ZORO_sjlt,np.mean(opt_sjlt.fd)] )
    # print some useful values
    opt_sjlt.report( 'Estimated f(x_k) using ZORO-sjlt: %f  function evals: %d\n' %
        (np.mean(opt_sjlt.fd), evals_ZORO_sjlt) )

fig, ax = plt.subplots()

ax.plot(np.array(performance_log_ZORO)[:,0],
 np.log10(np.array(performance_log_ZORO)[:,1]), linewidth=1, label = "ZORO")
ax.plot(np.array(performance_log_ZORO_sjlt)[:,0],
 np.log10(np.array(performance_log_ZORO_sjlt)[:,1]), linewidth=1, label = "ZORO with sjlt")
plt.xlabel('function evaluations')
plt.ylabel('$log($f(x)$)$')
leg = ax.legend()
plt.savefig('results/sjlt-vs-rademacher-on-maxk.png')