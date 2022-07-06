# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:56:42 2022

@author: nkyos
"""
import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

# Pars

omega = 1000

P0 = 10
h = 4.1

tau = 18.7 # unit: minutes
am = ap = 1
mum = mup = 0.03 # units: 1/min

stoptime = 120*60 # unit: minutes

sample_rate = 10 # try w 1
numpoints = int(stoptime/sample_rate)

past = np.array([0.2, 20], dtype = np.int32)


# =============================================================================
# Deterministic system
# =============================================================================


def hill(xt, x0, h):
    
    res = 1 / (1 + (xt/x0)**h)
    
    return res

# the equation
#y(0) = M, y(1) = P
f = [    
    am * hill( y( 1, t-tau ), P0, h) - mum * y(0, t),
    ap * y(0, t) - mup * y(1,t)
    ]

# initialising the integrator
DDE = jitcdde(f)

# enter initial conditions
DDE.constant_past(past)

# short pre-integration to take care of discontinuities
DDE.step_on_discontinuities()

# create timescale
times = DDE.t + np.linspace(1, stoptime, numpoints)

# integrating
data = np.zeros((times.size,2))
# data[0,:] = past
for i,time in enumerate(times):
    data[i,:] = DDE.integrate(time)


ax1lims = (20*60, 120*60) # hrs to minutes
ax1inds = np.where((times > ax1lims[0]) & (times < ax1lims[1]))[0]
times1, data1 = times[ax1inds]/60, data[ax1inds, :] # convert times to hrs


# fig, ax = plt.subplots(2)

# ax[0].plot(times1, data1[:,0], linewidth = 1, color = 'k')
# ax[0].set_ylabel('hes 1 mRNA')
# ax[0].set_ylim([0.5, 1])

# ax[1].plot(times1, data1[:,1], linewidth=1, color = 'k')
# ax[1].set_xlabel('time (h)')
# ax[1].set_ylabel('Hes 1 protein')
# ax[1].set_ylim([21, 30])

# plt.show()


# =============================================================================
# Stochastic system
# =============================================================================

from funcs_njit import rt_delay_gm

sr = rt_delay_gm(omega, P0, h, tau, am, ap, mum, mup,
                 tmax=stoptime, K=numpoints, past=past)
sr1 = sr[ax1inds, :]

fig, ax = plt.subplots(2)

ax[0].plot(times1, data1[:,0], linewidth = 1, color = 'k')
ax[0].plot(times1, sr1[:,0]/omega, linewidth = 1, color = 'red')
ax[0].set_ylabel('hes 1 mRNA')
# ax[0].set_ylim([0.5, 1])

ax[1].plot(times1, data1[:,1], linewidth=1, color = 'k')
ax[1].plot(times1, sr1[:,1]/omega, linewidth = 1, color = 'blue')
ax[1].set_xlabel('time (h)')
ax[1].set_ylabel('Hes 1 protein')
# ax[1].set_ylim([21, 30])

plt.show()

# =============================================================================
# Power spetrum
# =============================================================================

# import math
# from funcs_njit import rt_delay_gm_ps
# from scipy.fft import fftfreq



# omega = 500
# M = 1
# numpoints = stoptime
# eqt = 3000
# N = numpoints-eqt+1

# ps = rt_delay_gm_ps(omega, P0, h, tau, am, ap, mum, mup, tmax=stoptime, K=numpoints, past=past, M=M, eqt=eqt)



# fig, ax = plt.subplots()

# ax.plot(ps[:,1], color='red', marker = 'o',linestyle='none', fillstyle = 'none',
#             markersize = 3)

# plt.yscale('log')
# plt.xlabel(r'$\omega$')
# plt.ylabel(r'$P(\omega)$')