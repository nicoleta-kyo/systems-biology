# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:00:58 2022

@author: nkyos
"""

import numpy as np 
from numba import njit
from scipy.fft import fft, rfft

@njit
def my_digitise_right(x, bins):
     
     bins_size = bins.shape[0]
     
     if x <= bins[0]:
         return 0
     
     for i in range(1, bins_size):
         if ((x > bins[i-1]) and (x <= bins[i])):
             return i
     
     return bins_size+1    # return out of range value
 
@njit    
def get_delayed_reaction_index(arr, t1, t2):
    '''
    
    What if more than one?
    
    Parameters:
    ----------
    arr : numpy array 1-dim
        the array with times of the delayed reactions
    '''
    
    for ind,el in enumerate(arr):
        if (el >= t1):
            if (el < t2): 
                return ind 

    return -1
    

@njit
def rt_delay(a,b,c,tau,omega, tmax, n0, K):
    """
    Residence time (Gillespie) process simulation with delay
        for the toy model in arXiv:0901.3271v1
    Returns the avg n(t) and its variance
    
    Single trajectory!

    Parameters
    ----------
    pars : list
        list with rates a, b, c, the delay tau and the system size omega
    tmax : int
        max time for simulation.
    n0 : int
        initial number of particles
    K : int
        number of temporal bins of size h=tmax/K; for example maxt=100 time steps (dt=0.1)
                                                    for K=200, h = 100/200 = 0.5 
        
    Returns
    -------
    x : array
        the avg n for each bin k, as well as its variance

    """
    
    h = tmax/K
    x = np.zeros((K+1,2)) # store num of particles n as well as n**2 for the variance
        
    k0 = 1 # vars for saving the results
    t1 = 0
    
    tdr = np.empty(shape=1, dtype = np.float32) # times for delayed reactions list
    tdr[0] = -1
    
    n = n0 # n is current number of particles
    x[0,0] = n # store initial number of particles
    while (t1 < tmax):
        
        #--  res time algo
        a1 = a*omega
        a2 = b*n
        a3 = c*n
        a0 = a1 + a2 + a3
        
        tn = -np.log(np.random.random())/a0            
            
        dind = get_delayed_reaction_index(tdr[:10], t1, t1+tn)
        
        if dind >= 0:  # if delay
            if n > 0:
                n -= 1
            
            t0 = t1               
            t1 = tdr[dind]
            tdr = np.delete(tdr, dind)    # what if more than one index?
        
        else:   # if no delay # dummy
            
            r = np.random.random()*a0
            bins = np.cumsum(np.array([0,a1,a2,a3]))
            o = my_digitise_right(r, bins)

            if o == 1:
                n += 1
            elif o == 2:
                n -= 1
            else: #schedule delayed reaction
                arr = np.zeros(1, dtype=np.float32)
                arr[0] = t1+tau
                tdr = np.concatenate((tdr, arr))
                
            t0=t1
            t1=t0+tn  
        
        #--
        
       
        
        #--  store the result in the bins

        t1=tmax if (t1>tmax) else t1
        k1=int(t1/h)+1
        if (k1==k0):
            xc = (t1-t0)*n # current x val
            x[k0, 0] += xc
            # x[k0, 1] += xc**2
        else:
            xc = (k0*h-t0)*n
            x[k0, 0] += xc
            # x[k0, 1] += xc**2
            for k in range(k0+1,k1-1):
                xc = h*n
                x[k, 0] += xc
                # x[k, 1] += xc**2
            xc = (t1-(k1-1)*h)*n
            x[k1, 0] += xc
            # x[k1, 1] += xc**2
            
        k0=k1     

    x /= h        
    
    return x

@njit
def rt_delay_av(a,b,c,tau,omega, tmax, n0, K, M):
    """
    Residence time (Gillespie) process simulation with delay
        for the toy model in arXiv:0901.3271v1
    Returns the avg n(t) and its variance
    
    Power spectrim

    Parameters
    ----------
    pars : list
        list with rates a, b, c, the delay tau and the system size omega
    tmax : int
        max time for simulation.
    n0 : int
        initial number of particles
    M : int
        MC steps
    K : int
        number of temporal bins of size h=tmax/K; for example maxt=100 time steps (dt=0.1)
                                                    for K=200, h = 100/200 = 0.5 
        
    Returns
    -------
    x : array
        the avg n for each bin k, as well as its variance

    """
    
    h = tmax/K
    xM = np.zeros((K+1,2))
    
    for i in range(M):
        
        x = np.zeros((K+1,2)) # store num of particles n as well as n**2 for the variance
            
        k0 = 1 # vars for saving the results
        t1 = 0
        
        tdr = np.empty(shape=1, dtype = np.float32) # times for delayed reactions list
        tdr[0] = -1
        
        n = n0 # n is current number of particles
        x[0,0] = n # store initial number of particles
        while (t1 < tmax):
            
            #--  res time algo
            a1 = a*omega
            a2 = b*n
            a3 = c*n
            a0 = a1 + a2 + a3
            
            tn = -np.log(np.random.random())/a0            
                
            dind = get_delayed_reaction_index(tdr[:10], t1, t1+tn)
            
            if dind >= 0:  # if delay
                if n > 0:
                    n -= 1
                
                t0 = t1               
                t1 = tdr[dind]
                tdr = np.delete(tdr, dind)    # what if more than one index?
            
            else:   # if no delay # dummy
                
                r = np.random.random()*a0
                bins = np.cumsum(np.array([0,a1,a2,a3]))
                o = my_digitise_right(r, bins)

                if o == 1:
                    n += 1
                elif o == 2:
                    n -= 1
                else: #schedule delayed reaction
                    arr = np.zeros(1, dtype=np.float32)
                    arr[0] = t1+tau
                    tdr = np.concatenate((tdr, arr))
                    
                t0=t1
                t1=t0+tn  
            
            #--
            
           
            
            #--  store the result in the bins

            t1=tmax if (t1>tmax) else t1
            k1=int(t1/h)+1
            if (k1==k0):
                xc = (t1-t0)*n # current x val
                x[k0, 0] += xc
                # x[k0, 1] += xc**2
            else:
                xc = (k0*h-t0)*n
                x[k0, 0] += xc
                # x[k0, 1] += xc**2
                for k in range(k0+1,k1-1):
                    xc = h*n
                    x[k, 0] += xc
                    # x[k, 1] += xc**2
                xc = (t1-(k1-1)*h)*n
                x[k1, 0] += xc
                # x[k1, 1] += xc**2
                
            k0=k1   
            
        x /= h 
        xM += x

    # averaging    
    xM /= M
    
    return xM


def rt_delay_ps(a,b,c,tau,omega, tmax, n0, K, M, eqt):
    """
    Residence time (Gillespie) process simulation with delay
        for the toy model in arXiv:0901.3271v1
    Returns the avg n(t) and its variance
    
    Power spectrim

    Parameters
    ----------
    pars : list
        list with rates a, b, c, the delay tau and the system size omega
    tmax : int
        max time for simulation.
    n0 : int
        initial number of particles
    M : int
        MC steps
    K : int
        number of temporal bins of size h=tmax/K; for example maxt=100 time steps (dt=0.1)
                                                    for K=200, h = 100/200 = 0.5 
        
    Returns
    -------
    x : array
        the avg n for each bin k, as well as its variance

    """
    
    xst = a/(b+c) # x star
    
    for i in range(M):
        
        print(i)
        print('Calc x')
        # x = rt_delay(a,b,c,tau, omega, tmax, n0, K)
        x = rt_delay(a,b,c,tau, omega, tmax, n0, K)[eqt:,:]
        
        # calculate the fluctuations and calc the FFT of Xi
        xi = np.sqrt(omega)*( (x[:,0]/omega) - xst)
        
        print('Calc fft')
        xi_fft = fft(xi)
        
        if i == 0:
            axifs = np.abs(xi_fft)**2
        else:
            axifs += np.abs(xi_fft)**2
        
    # averaging    
    axifs /= M
    
    return axifs


@njit
def hill(xt, x0, h):
    
    res = 1 / (1 + (xt/x0)**h)
    
    return res


@njit
def rt_delay_gm(omega, P0, h, tau, am, ap, mum, mup, tmax, K, past):
    """
    Residence time (Gillespie) process simulation with delay
        for the gene expression model in arXiv:0901.3271v1
    One run!

    Parameters
    ----------
    omega, P0, h, tau, am, ap, mum, mup : 
        all the model parameters: the system size omega, P0 and h for the hill func,
        the delay tau, the growth rates alpha_M and alpha_P and the death rates mu_M and mu_P
    tmax : int
        max time for simulation.
    K : int
        number of temporal bins of size h=tmax/K; for example maxt=100 time steps (dt=0.1)
                                                    for K=200, h = 100/200 = 0.5 
    past : numpy array of size 2
        the initial numbers of M and P
        
    Returns
    -------
    x : array of shape K, 2
        n_M and n_P at each bin K

    """
    
    bin_size = tmax/K
    x = np.zeros((K+1,2)) # store num of particles M and P
        
    k0 = 1 # vars for saving the results
    t1 = 0
    
    tdr = np.empty(shape=1, dtype = np.float32) # times for delayed reactions list
    tdr[0] = -1
    
    m = past[0] # m and p is current number of particles
    p = past[1]
    
    # x[0,:] = past # store initial number of particles
    while (t1 < tmax):
        
        #--  res time algo
        a1 = mum*m
        a2 = mup*p
        a3 = ap*m
        a4 = omega*am*hill(p/omega, P0, h)
        a0 = a1 + a2 + a3 + a4
        
        tn = -np.log(np.random.random())/a0            
            
        dind = get_delayed_reaction_index(tdr[:10], t1, t1+tn)
        
        if dind >= 0:  # if delay
            m += 1
            
            t0 = t1               
            t1 = tdr[dind]
            tdr = np.delete(tdr, dind)    # what if more than one index?
        
        else:   # if no delay
            
            r = np.random.random()*a0
            bins = np.cumsum(np.array([0,a1,a2,a3,a4]))
            o = my_digitise_right(r, bins)

            if o == 1:
                if m > 0:
                    m -= 1
            elif o == 2:
                if p > 0:
                    p -= 1
            elif o == 3:
                p += 1
            else: #schedule delayed reaction
                arr = np.zeros(1, dtype=np.float32)
                arr[0] = t1+tau
                tdr = np.concatenate((tdr, arr))
                
            t0=t1
            t1=t0+tn  
        
        #--
        
        #--  store the result in the bins
        n = np.array([m,p], dtype = np.float32)
        
        t1=tmax if (t1>tmax) else t1
        # k1=int(t1/bin_size)+1
        k1=int(t1/bin_size)+1
        if (k1==k0):
            xc = (t1-t0)*n # current x val
            x[k0-1, :] += xc
            # x[k0, 1] += (t1-t0)*(n**2)
        else:
            xc = (k0*bin_size-t0)*n
            x[k0-1, :] += xc
            # x[k0, 1] += (k0*bin_size-t0)*(n**2)
            # for k in range(k0+1,k1-1):
            for k in range(k0+1, k1-1):    
                xc = bin_size*n
                x[k-1, :] += xc
                # x[k, 1] += bin_size*(n**2)
            # xc = (t1-(k1-1)*bin_size)*n
            xc = (t1 - ((k1-1)*bin_size) ) * n
            x[k1-1, :] += xc
            # x[k1, 1] += (t1-(k1-1)*bin_size)*(n**2)
            
        k0=k1     

    x /= bin_size       
    
    return x


def rt_delay_gm_ps(omega, P0, h, tau, am, ap, mum, mup, tmax, K, past, M, eqt):
    """
    Residence time (Gillespie) process simulation with delay
        for the gene model in arXiv:0901.3271v1
    Returns the avg n(t) and its variance
    
    Power spectrim

    Parameters
    ----------
    pars : list
        list with rates a, b, c, the delay tau and the system size omega
    tmax : int
        max time for simulation.
    n0 : int
        initial number of particles
    M : int
        MC steps
    K : int
        number of temporal bins of size h=tmax/K; for example maxt=100 time steps (dt=0.1)
                                                    for K=200, h = 100/200 = 0.5 
        
    Returns
    -------
    x : array
        the avg n for each bin k, as well as its variance

    """
    
    mst = 0.7522
    pst = 25.072
    
    axifs = np.zeros((K-eqt+1,2))
    for i in range(M):
        
        print(i)
        print('Calc x')
        x = rt_delay_gm(omega, P0, h, tau, am, ap, mum, mup, tmax, K, past)[eqt:,:]
        
        # calculate the fluctuations and calc the FFT of Xi
        m_xi = np.sqrt(omega)*(x[:,0]/omega - mst)
        p_xi = np.sqrt(omega)*(x[:,1]/omega - pst)
        
        print('Calc fft')
        m_xi_fft = fft(m_xi)
        p_xi_fft = fft(p_xi)
        
        axifs[:,0] += np.abs(m_xi_fft.astype(np.float64))**2
        axifs[:,1] += np.abs(p_xi_fft.astype(np.float64))**2
        
    # averaging    
    axifs /= M
    
    return axifs