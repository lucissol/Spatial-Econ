# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:07:53 2026

@author: lseibert
"""

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def monte_carlo_gini(model, r, w_path, H, T, S=50):
    """
    S: Number of simulations (trials)
    """
    all_paths = []
    
    for s in range(S):
        # We need a new seed or just let it use the global random state
        path = model.simulate_gini_path(r, w_path, H=H, T=T)
        all_paths.append(path)
        
    all_paths = np.array(all_paths)
    mean_path = np.mean(all_paths, axis=0)
    std_path = np.std(all_paths, axis=0)
    
    return mean_path, std_path

def monte_carlo_gini_parallel(model, r, w_path, H, T, a_steady=None, S=50, n_jobs=-1):
    """
    Runs S Monte Carlo simulations in parallel.
    
    Parameters:
    -----------
    n_jobs : int
        Number of CPU cores to use. -1 uses all available cores.
        -2 uses all but one.
    """
    print(f"Starting {S} simulations on {n_jobs if n_jobs > 0 else 'all'} cores...")

    # THE PARALLEL LOOP
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(model.simulate_gini_path)(
            r, 
            w_path,      # The wage path
            initial_dist=a_steady, # Start from steady state
            seed=s,           # Unique seed
            H=10_000,         # Agents per sim
            T=100             # Time periods
        ) 
        for s in range(100)   # 100 Simulations
    )
    
    # Aggregation (The "Reduce" step)
    # results is a list of arrays. We stack them into a matrix (S x T)
    all_paths = np.array(results)
    
    mean_path = np.mean(all_paths, axis=0)
    std_path = np.std(all_paths, axis=0)
    
    return mean_path, std_path

def plot_spatial_lorenz(a_core, a_periphery, b):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    def get_lorenz_coords(assets):
        # 1. Sort and shift to wealth space (non-negative)
        y = np.sort(assets + b)
        n = len(y)
        
        # 2. Cumulative shares
        cum_wealth = np.cumsum(y) / np.sum(y)
        cum_pop = np.arange(1, n + 1) / n
        
        # 3. Force (0,0) starting point
        return np.append([0], cum_pop), np.append([0], cum_wealth)

    # Calculate coordinates
    x_core, y_core = get_lorenz_coords(a_core)
    x_peri, y_peri = get_lorenz_coords(a_periphery)
    
    # Plotting
    ax.plot(x_core, y_core, label='Core Region', color='steelblue', lw=2.5)
    ax.plot(x_peri, y_peri, label='Periphery Region', color='indianred', lw=2.5)
    
    # 45-degree line (Perfect Equality)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.6, label='Perfect Equality')
    
    # Formatting
    ax.set_title("Spatial Inequality: Lorenz Curves", fontsize=15, fontweight='bold')
    ax.set_xlabel("Cumulative Share of Population", fontsize=12)
    ax.set_ylabel("Cumulative Share of Wealth", fontsize=12)
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    ax.set_aspect('equal') # Keep it square
    
    plt.tight_layout()
    plt.show()