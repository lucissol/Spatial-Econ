# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 14:00:13 2026

@author: lseibert
"""

class config:
    # Macro Layer
    T             = 100
    rho_core      = 0.8
    rho_periphery = 0.6
    rho_c_p       = 0.3
    rho_p_c       = 0.05
    
    w_core = 1
    w_periphery = 0.8
    sigma = 1
    theta = 0.6
    mu_z = 0
    sigma_eps = 0.2
    
    # invariant distribution
    H_sim         = 25_000
    T_sim         = 1000
    burn_in       = 400
    
    # Monte Carlo Bootstrap
    H_na          = 20_000
    S_na          = 400
        
    # household
    beta          = 0.96
    r             = ((1/beta) - 1) * 0.95
    b = 0
    na = 1000