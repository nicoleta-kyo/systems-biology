# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:58:12 2022

@author: nkyos
"""

import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t


# Pars

a = 100
b = 4.1
c = 4
tau = 2

omega = 100 
x0 = 100
stoptime = 500.0


# =============================================================================
# Deterministic system
# =============================================================================

numpoints = 650
x_past = 50

# # the equation
# f = [    
#     a - b * y(0, t) - c * y(0, t-tau)
#     ]

# # initialising the integrator
# DDE = jitcdde(f)

# # enter initial conditions
# DDE.constant_past([x_past])

# # short pre-integration to take care of discontinuities
# DDE.step_on_discontinuities()

# # create timescale
# times = DDE.t + np.linspace(1, stoptime, numpoints)

# # integrating
# data = np.array([])
# for time in times:
#     data = np.append(data, DDE.integrate(time) )

# ax1lims = (75,250)
# ax2lims = (380, 420) 
# ax1inds = np.where((times > ax1lims[0]) & (times < ax1lims[1]))[0]
# ax2inds = np.where((times > ax2lims[0]) & (times < ax2lims[1]))[0]
# times1, data1 = times[ax1inds], data[ax1inds]
# times2, data2 = times[ax2inds], data[ax2inds]

# fig, ax = plt.subplots()

# left, bottom, width, height = [0.25, 0.6, 0.55, 0.2]
# ax2 = fig.add_axes([left, bottom, width, height])

# ax.plot(times1, data1, color='blue', linewidth=1)
# # ax.plot(times1, data1, color='blue', linewidth=1)
# ax.set_ylim([0, 40])
# ax.set_xlabel(r'$t$')
# ax.set_ylabel(r'$n/\Omega$')
# ax2.plot(times2, data2, color='blue', linewidth=1)
# ax2.set_ylim([11,14])

# # ax.set_title('$ihf(t)=e^t$')
# plt.show()



# # =============================================================================
# # Stochastic system - multiple runs
# # =============================================================================
 
# from funcs_njit import rt_delay_av

# seed = 123
# rgn = np.random.default_rng(seed)

# numpoints = 1500
# M = 100
# sr = rt_delay_av(a,b,c,tau, omega, tmax=stoptime, n0=x0, K=numpoints, M=M)



# fig, ax = plt.subplots()

# left, bottom, width, height = [0.25, 0.6, 0.55, 0.2]
# # ax2 = fig.add_axes([left, bottom, width, height])

# # ax1lims = (50,250)
# # ax1lims = (50, stoptime) # for testing
# times_stoch = np.linspace(start=0, stop=stoptime, num=numpoints)

# data1inds_stoch = np.where((times_stoch > ax1lims[0]) & (times_stoch < ax1lims[1]))[0]
# data2inds_stoch = np.where((times_stoch > ax2lims[0]) & (times_stoch < ax2lims[1]))[0]

# times1_stoch, data1_stoch = times_stoch[data1inds_stoch], sr[data1inds_stoch, 0]
# times2_stoch, data2_stoch = times_stoch[data2inds_stoch], sr[data2inds_stoch, 0]

# ax.plot(times1_stoch, data1_stoch/omega, color='red', linewidth=1) 
# # ax.plot(times1, data1, color='blue', linewidth=1) # the deterministic system

# # ax.set_ylim([7.5, 25])

# ax.set_xlabel(r'$t$')
# ax.set_ylabel(r'$n/\Omega$')

# # ax2.plot(times2, data2, color='blue', linewidth=1) # the inset plot deterministic
# # ax2.plot(times2_stoch, data2_stoch/omega, color='red', linewidth = 1) # inset stoch
# # ax2.set_ylim([9,15])

# plt.show()


# =============================================================================
# Stochastic system
# =============================================================================
 

# from funcs_njit import rt_delay_2


# # stoptime = 500 #250 # / 0.77 to get numpoints like above

# omega = 100 
# numpoints = 1500
# sr = rt_delay_2(a,b,c,tau, omega, tmax=stoptime, n0=x0, K=numpoints)



# fig, ax = plt.subplots()

# left, bottom, width, height = [0.25, 0.6, 0.55, 0.2]
# ax2 = fig.add_axes([left, bottom, width, height])

# # ax1lims = (50,250)
# # ax1lims = (50, stoptime) # for testing
# times_stoch = np.linspace(start=0, stop=stoptime, num=numpoints)

# data1inds_stoch = np.where((times_stoch > ax1lims[0]) & (times_stoch < ax1lims[1]))[0]
# data2inds_stoch = np.where((times_stoch > ax2lims[0]) & (times_stoch < ax2lims[1]))[0]

# times1_stoch, data1_stoch = times_stoch[data1inds_stoch], sr[data1inds_stoch, 0]
# times2_stoch, data2_stoch = times_stoch[data2inds_stoch], sr[data2inds_stoch, 0]

# ax.plot(times1_stoch, data1_stoch, color='red', linewidth=1) 
# ax.plot(times1, data1, color='blue', linewidth=1) # the deterministic system

# # ax.set_ylim([7.5, 25])

# ax.set_xlabel(r'$t$')
# ax.set_ylabel(r'$n/\Omega$')

# ax2.plot(times2, data2, color='blue', linewidth=1) # the inset plot deterministic
# ax2.plot(times2_stoch, data2_stoch, color='red', linewidth = 1) # inset stoch
# # ax2.set_ylim([9,15])

# plt.show()


# =============================================================================
# Power spectrum
# =============================================================================

import math
from funcs_njit import rt_delay_ps_2
from scipy.fft import fftfreq


def calc_ps(ws,a,b,c, tau):
    
    wsn = np.zeros_like(ws)
    for i in range(wsn.shape[0]):
        wsn[i] = (2*a) / ( (b + c*math.cos(ws[i]*tau))**2 + (ws[i] - c*math.sin(ws[i]*tau))**2 )
        
    return wsn
    
#

x0 = 400 #? 
omega = 200 #200
M = 1
numpoints = 5000
eqt = 100
N = numpoints-eqt+1

# analytical

ws = np.linspace(start=0, stop=stoptime, num=N)
anps = calc_ps(ws, a, b, c, tau)
plt.plot(ws[:150], anps[:150])
plt.ylabel('$P(\omega)$')
plt.xlabel('$\omega$')
plt.yscale('log')


# numerical 
# numpoints = 1000
ps = rt_delay_ps_2(a,b,c,tau,omega,tmax=stoptime, n0=x0, K=numpoints, M=M, eqt=eqt)

freq = fftfreq(n=N,d=1/(N/stoptime)) # OR 1
# freq = [i/numpoints for i in range(numpoints)] # OR 2
#

fig, ax = plt.subplots()

ax.plot(freq[:N//2], ps[:N//2], color='red', marker = 'o',linestyle='none', fillstyle = 'none',
            markersize = 3)
# ax.plot(freq[(freq >= 0) & (freq<2.25)], ps[(freq >= 0) & (freq<2.25)], color='red', marker = 'o',linestyle='none', fillstyle = 'none',
            # markersize = 3)

# ax2 = ax.twinx()  # instantiate a second axes for the times
# ax2.plot(anps[:150], color = 'blue')

plt.yscale('log')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$P(\omega)$')


